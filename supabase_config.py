from supabase import create_client
import os
from dotenv import load_dotenv
from typing import Dict, Optional
from fastapi import HTTPException

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    raise ValueError("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY in .env file")

supabase = create_client(supabase_url, supabase_key)

class Auth:
    @staticmethod
    async def sign_up(email: str, password: str) -> Dict:
        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def sign_in(email: str, password: str) -> Dict:
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return response
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    @staticmethod
    async def sign_out(token: str) -> Dict:
        try:
            response = supabase.auth.sign_out()
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def get_user(token: str) -> Optional[Dict]:
        try:
            response = supabase.auth.get_user(token)
            return response
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid token")

    @staticmethod
    async def reset_password(email: str) -> Dict:
        try:
            response = supabase.auth.reset_password_for_email(email)
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def update_password(token: str, new_password: str) -> Dict:
        try:
            response = supabase.auth.update_user({
                "password": new_password
            })
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) 