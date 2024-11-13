from flask import Flask
from flask_cors import CORS
from routes import bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8445, ssl_context=('/home/mt798jx/chatbot/cert.pem', '/home/mt798jx/chatbot/key.pem'))