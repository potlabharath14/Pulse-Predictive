from pymongo import MongoClient
import gridfs
from flask_login import UserMixin
from bson.objectid import ObjectId

# ─── MongoDB Connection ─────────────────────────────────────────────────────
client = MongoClient('mongodb://localhost:27017/pulse_predictive')
mongo_db = client['pulse_predictive']
users_col = mongo_db['users']
records_col = mongo_db['records']
reports_col = mongo_db['reports']
fs = gridfs.GridFS(mongo_db)

users_col.create_index('username', unique=True)
records_col.create_index('user_id')
reports_col.create_index('prediction_id')

class MongoUser(UserMixin):
    def __init__(self, user_doc):
        self.user_doc = user_doc
        self.id = str(user_doc['_id'])
        self.username = user_doc['username']

def get_user_by_id(user_id):
    try:
        doc = users_col.find_one({'_id': ObjectId(user_id)})
        if doc:
            return MongoUser(doc)
    except Exception:
        pass
    return None
