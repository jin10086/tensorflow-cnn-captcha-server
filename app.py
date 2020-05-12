from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_pymongo import PyMongo

import werkzeug, os
import uuid
from model import test_train

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/captcha"
mongo = PyMongo(app)
api = Api(app)


UPLOAD_FOLDER = "wait_train_images"
parser = reqparse.RequestParser()
parser.add_argument("file", type=werkzeug.datastructures.FileStorage, location="files")

parser1 = reqparse.RequestParser()
parser1.add_argument("reqid", type=str)
parser1.add_argument("status", type=str)


class PhotoUpload(Resource):
    decorators = []

    def post(self):
        data = parser.parse_args()
        if data["file"] == "":
            return {"data": "", "message": "No file found", "status": "error"}
        photo = data["file"]
        if photo:
            save_2_name = os.path.join(UPLOAD_FOLDER, photo.filename)
            photo.save(save_2_name)
            label = test_train(save_2_name)
            reqid = str(uuid.uuid4())
            mongo.db.captcha.insert_one({"status": 0, "label": label, "reqid": reqid})
            return {"label": label, "reqid": reqid, "filename": save_2_name}
        return {"data": "", "message": "Something when wrong", "status": "error"}

    def get(self):
        # http://127.0.0.1:5000/upload\?status\=1\&reqid\=2
        data = parser1.parse_args()
        reqid = data["reqid"]
        if int(data["status"]) == 1:  # 验证码正确
            q = mongo.db.captcha.find_one({"reqid": reqid})
            if q:
                q["status"] = 1
                mongo.db.captcha.update_one({"reqid": reqid}, {"$set": q})
                return {"ok": 1}
        return {"ok": 0}


api.add_resource(PhotoUpload, "/upload")

if __name__ == "__main__":
    app.run(debug=True)
