from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase_config import Auth
from typing import Optional

security = HTTPBearer()

class AuthMiddleware:
    def __init__(self):
        self.security = security

    async def __call__(self, request: Request) -> Optional[dict]:
        try:
            # Get the authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                raise HTTPException(status_code=401, detail="No authorization header")

            # Extract the token
            token = auth_header.split(' ')[1]
            if not token:
                raise HTTPException(status_code=401, detail="Invalid token format")

            # Verify the token and get user
            user = await Auth.get_user(token)
            if not user:
                raise HTTPException(status_code=401, detail="Invalid token")

            # Add user to request state
            request.state.user = user
            return user

        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

# Create middleware instance
auth_middleware = AuthMiddleware() 