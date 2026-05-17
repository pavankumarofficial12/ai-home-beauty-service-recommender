# Home Service Provider - Personalized Beauty & Haircare Recommendations

A smart backend system that helps users discover the best beauty and haircare services at home. It analyzes the user's skin, hair, and scalp profile and recommends suitable services from nearby providers using AI.

---

## About This Project

This API was built to solve a real problem — finding the right beauty/hair service that matches a person's unique profile (skin type, hair type, dandruff issues, face shape, etc.) instead of just showing random nearby salons.

It combines **location intelligence**, **user profile matching**, and **AI judgment** (powered by Llama via AWS Bedrock) to deliver highly relevant recommendations.

---

## Key Features

- Finds service providers within 200 km radius
- Uses AI to check whether a service is safe and suitable for the user
- Considers skin type, skin tone, hair type, scalp condition, gender, and face type
- Prioritizes services that the user specifically requests
- Sorts results intelligently by rating and distance
- Fast performance using parallel processing

---

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL (Separate DBs for users and service providers)
- **AI Model**: AWS Bedrock - Llama 3
- **Location Engine**: Haversine formula
- **Others**: boto3, psycopg2, Pydantic

---

## How to Set Up Locally

### 1. Clone the Project
```bash
git clone https://github.com/yourusername/home-service-provider.git
cd home-service-provider
