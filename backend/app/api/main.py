from fastapi import APIRouter,FastAPI
from app.api.routes import items, login, users, utils, predict
#from app.api import api_router  # Adjust the import path according to your structure

#app = FastAPI()
# Include the API router with versioning
#app.include_router(api_router, prefix="/api/v1")

api_router = APIRouter()

api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
