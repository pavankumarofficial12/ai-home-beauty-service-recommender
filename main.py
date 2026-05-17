
import logging
import json
from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Optional, Any

import boto3
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import psycopg2
from psycopg2.extras import DictCursor
import uvicorn

# ────────────────────────────────────────────────
#                IMPORT SETTINGS FROM .env / settings.py
# ────────────────────────────────────────────────
from config.settings import (
    DB_HOST,
    DB_USER,
    DB_PASSWORD,
    DB_PORT,
    DB_NAME2,
    DB_NAME3,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    BEDROCK_MODEL_ID,
)

# ────────────────────────────────────────────────
#                   LOGGING SETUP
# ────────────────────────────────────────────────
logger = logging.getLogger("home-service-provider")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)s | %(message)s')
)
logger.addHandler(console_handler)

logger.info("=== HOME SERVICE PROVIDER FastAPI APPLICATION STARTED ===")
logger.info(f"Bedrock Model ID: {BEDROCK_MODEL_ID}")
logger.info(f"Users DB: {DB_NAME2} | Service Providers DB: {DB_NAME3}")
logger.info(f"Database host: {DB_HOST}  port: {DB_PORT}")

# ────────────────────────────────────────────────
#                   FASTAPI APP
# ────────────────────────────────────────────────
app = FastAPI(
    title="Home Service Provider - Personalized Recommendations",
    description="Find suitable beauty/hair services near user with personalization",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ────────────────────────────────────────────────
#                   Pydantic MODELS
# ────────────────────────────────────────────────
class UserRequest(BaseModel):
    user_id: int
    requested_services: Optional[List[str]] = None

class ServiceOut(BaseModel):
    service_type: str
    price: float
    name: str
    is_active: bool
    duration: int
    suitable: bool
    reason: str
    provider_id: int
    distance_km: float
    average_rating: Optional[float] = None
    is_requested: Optional[bool] = False
    artist_service_types: List[str] = []

class UserProfileOut(BaseModel):
    skin_type: Optional[str] = None
    skin_tone: Optional[str] = None
    face_type: Optional[str] = None
    gender: Optional[str] = None
    scalp_type: Optional[str] = None
    hair_type: Optional[str] = None
    dandruff_detected: Optional[bool] = None

class RecommendationResponse(BaseModel):
    user_id: int
    user_profile: UserProfileOut
    recommended_services: List[ServiceOut]
    requested_services_matched: List[ServiceOut]
    total_recommended: int
    max_radius_used_km: float
    sorting: str

# ────────────────────────────────────────────────
#                   HELPER FUNCTIONS
# ────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(R * c, 2)


def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )


# ──── MOVED HERE so Pylance can see it before it's called ─────
def check_suitability(
    service_check: Dict,
    user_profile: Dict,
    bedrock_client,
    model_id: str
) -> Dict:
    """Call Llama model to check if service is suitable for user"""
    service = service_check['service']
    prompt = f"""You are a professional beauty and haircare expert.

USER PROFILE:
- Skin Type     : {user_profile.get('skin_type', 'unknown')}
- Skin Tone     : {user_profile.get('skin_tone', 'unknown')}
- Face Type     : {user_profile.get('face_type', 'unknown')}
- Gender        : {user_profile.get('gender', 'unknown')}
- Scalp Type    : {user_profile.get('scalp_type', 'unknown')}
- Hair Type     : {user_profile.get('hair_type', 'unknown')}
- Dandruff      : {user_profile.get('dandruff_detected', 'unknown')}

SERVICE:
- Name          : {service.get('name', 'N/A')}
- Type          : {service.get('service_type', 'N/A')}

Is this service safe, suitable and recommended for this user?
Answer ONLY with valid JSON (nothing else):

{{"suitable": true or false, "reason": "one short sentence"}}
"""

    try:
        body = json.dumps({
            "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            "max_gen_len": 256,
            "temperature": 0.1,
            "top_p": 0.9,
        })

        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        result = json.loads(response["body"].read().decode())
        generation = result.get("generation", "")

        # Extract JSON block
        match = re.search(r'\{[\s\S]*?\}', generation)
        if match:
            parsed = json.loads(match.group())
            suitable = bool(parsed.get("suitable", False))
            reason = parsed.get("reason", "No explanation provided")
        else:
            suitable = False
            reason = "Could not parse model response"

        return {**service_check, "suitable": suitable, "reason": reason}

    except Exception as e:
        logger.error(f"Bedrock LLM call failed: {str(e)}")
        return {**service_check, "suitable": False, "reason": f"LLM error: {str(e)}"}


# ────────────────────────────────────────────────
#                   MAIN ENDPOINT
# ────────────────────────────────────────────────
@app.post("/recommend-services", response_model=RecommendationResponse)
async def recommend_home_services(request: UserRequest = Body(...)):
    logger.info(f"New request → user_id={request.user_id} | requested={request.requested_services}")

    users_conn = None
    sp_conn = None

    try:
        # ─── Database connections (PostgreSQL) ────────────────────────────────
        users_conn = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            dbname=DB_NAME2,
            connect_timeout=15,
            cursor_factory=DictCursor
        )

        sp_conn = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            dbname=DB_NAME3,
            connect_timeout=15,
            cursor_factory=DictCursor
        )

        # ─── 1. Fetch user profile + optional hair analysis ──────
        with users_conn.cursor() as cur:
            cur.execute("""
                SELECT user_id, latitude, longitude, skin_type, skin_tone,
                       face_type, gender, scalp_type
                FROM user_profiles
                WHERE user_id = %s
            """, (request.user_id,))
            profile_row = cur.fetchone()

        if not profile_row:
            raise HTTPException(404, detail=f"User {request.user_id} not found in user_profiles")

        profile = dict(profile_row)

        with users_conn.cursor() as cur:
            cur.execute("""
                SELECT hair_type, dandruff_detected
                FROM hair_analysis_results
                WHERE user_id = %s
            """, (request.user_id,))
            hair_row = cur.fetchone()

            if hair_row:
                profile.update(dict(hair_row))

        user_lat = profile.get('latitude')
        user_lon = profile.get('longitude')

        if user_lat is None or user_lon is None:
            raise HTTPException(400, detail="User location (latitude/longitude) is missing")

        # ─── 2. Find nearby providers (≤ 200 km) ─────────────────
        MAX_RADIUS_KM = 200.0

        with sp_conn.cursor() as cur:
            cur.execute("SELECT id, salon_id, latitude, longitude FROM service_providers")
            all_providers = cur.fetchall()

        nearby = []
        for p in all_providers:
            if p['latitude'] is None or p['longitude'] is None:
                continue
            dist = haversine(user_lat, user_lon, p['latitude'], p['longitude'])
            if dist <= MAX_RADIUS_KM:
                nearby.append({**dict(p), 'distance_km': dist})

        if not nearby:
            logger.info("No providers found within radius")
            return RecommendationResponse(
                user_id=request.user_id,
                user_profile=UserProfileOut(**profile),
                recommended_services=[],
                requested_services_matched=[],
                total_recommended=0,
                max_radius_used_km=MAX_RADIUS_KM,
                sorting="rating (desc) → distance (asc)"
            )

        # ─── 3. Collect services, ratings, artist types from artists table ─────
        bedrock = get_bedrock_client()
        service_checks = []

        with sp_conn.cursor() as cur:
            for prov in nearby:
                pid = prov['id']
                dist = prov['distance_km']

                # Services
                cur.execute("""
                    SELECT service_type, price, name, is_active, duration
                    FROM services
                    WHERE service_provider_id = %s
                """, (pid,))
                services = cur.fetchall()

                # Average rating
                cur.execute("""
                    SELECT AVG(rating) as avg_rating
                    FROM ratings
                    WHERE service_provider_id = %s
                """, (pid,))
                rating_row = cur.fetchone()
                avg_rating = round(rating_row['avg_rating'], 2) if rating_row['avg_rating'] else None

                # Artist service types (table name: artists)
                cur.execute("""
                    SELECT service_type
                    FROM artists
                    WHERE service_provider_id = %s
                """, (pid,))
                artist_rows = cur.fetchall()
                artist_service_types = [row['service_type'] for row in artist_rows]

                for svc in services:
                    if svc.get('is_active', False):
                        service_checks.append({
                            'provider_id': pid,
                            'distance_km': dist,
                            'average_rating': avg_rating,
                            'service': dict(svc),
                            'artist_service_types': artist_service_types
                        })

        logger.info(f"Evaluating {len(service_checks)} active services with LLM")

        # ─── 4. Parallel suitability checks ──────────
        suitable_services = []
        if service_checks:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        check_suitability,           # ← now defined above
                        item,
                        profile,
                        bedrock,
                        BEDROCK_MODEL_ID
                    )
                    for item in service_checks
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result.get('suitable'):
                        suitable_services.append(result)

        # Sort: highest rating first, then closest
        suitable_services.sort(
            key=lambda x: (-(x.get('average_rating') or 0), x['distance_km'])
        )

        # ─── 5. Match requested services (if provided) ──────────
        requested_matched = []
        if request.requested_services:
            for req in request.requested_services:
                req_lower = req.lower()
                for item in suitable_services:
                    name_lower = (item['service'].get('name') or '').lower()
                    type_lower = (item['service'].get('service_type') or '').lower()
                    if req_lower in name_lower or req_lower in type_lower:
                        copy_item = item.copy()
                        copy_item['is_requested'] = True
                        requested_matched.append(copy_item)
                        break

        # ─── Final structured response ──────────────────────────
        response = RecommendationResponse(
            user_id=request.user_id,
            user_profile=UserProfileOut(**{
                k: profile.get(k) for k in [
                    "skin_type", "skin_tone", "face_type", "gender",
                    "scalp_type", "hair_type", "dandruff_detected"
                ]
            }),
            recommended_services=[
                ServiceOut(
                    service_type=s['service']['service_type'],
                    price=s['service']['price'],
                    name=s['service']['name'],
                    is_active=s['service']['is_active'],
                    duration=s['service']['duration'],
                    suitable=s['suitable'],
                    reason=s['reason'],
                    provider_id=s['provider_id'],
                    distance_km=s['distance_km'],
                    average_rating=s['average_rating'],
                    is_requested=s.get('is_requested', False),
                    artist_service_types=s.get('artist_service_types', [])
                )
                for s in suitable_services
            ],
            requested_services_matched=[
                ServiceOut(
                    service_type=s['service']['service_type'],
                    price=s['service']['price'],
                    name=s['service']['name'],
                    is_active=s['service']['is_active'],
                    duration=s['service']['duration'],
                    suitable=s['suitable'],
                    reason=s['reason'],
                    provider_id=s['provider_id'],
                    distance_km=s['distance_km'],
                    average_rating=s['average_rating'],
                    is_requested=True,
                    artist_service_types=s.get('artist_service_types', [])
                )
                for s in requested_matched
            ],
            total_recommended=len(suitable_services),
            max_radius_used_km=MAX_RADIUS_KM,
            sorting="rating (desc) → distance (asc)"
        )

        logger.info(f"Success → {len(suitable_services)} recommended services")
        return response

    except HTTPException as e:
        raise e
    except psycopg2.OperationalError as db_err:
        logger.exception(f"Database connection error: {str(db_err)}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(db_err)}")
    except Exception as exc:
        logger.exception("Critical error in recommend-services endpoint")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if users_conn:
            users_conn.close()
        if sp_conn:
            sp_conn.close()


# ────────────────────────────────────────────────
#                   RUN SERVER
# ────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting uvicorn server on http://0.0.0.0:8000")
    logger.info("API documentation: http://127.0.0.1:8000/docs")
    uvicorn.run(
        "home_services:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
