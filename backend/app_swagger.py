from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
from functools import wraps
from models import User, Role, Video, PracticeSession, AnalyticsEvent
from bson import ObjectId

load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'fallback_secret_key')

# Swagger API setup
api = Api(
    app,
    version='1.0',
    title='ISL Learning Platform API',
    description='Complete API documentation for Indian Sign Language Learning Platform',
    doc='/docs/',
    prefix='/api/v1'
)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['thadomal_db']

# Collections
users_collection = db['users']
roles_collection = db['roles']
videos_collection = db['videos']
practice_sessions_collection = db['practice_sessions']
analytics_events_collection = db['analytics_events']
glyphs_collection = db['glyphs']
alphabet_collection = db['alphabet']
vocabulary_collection = db['vocabulary']
sentences_collection = db['sentences']
stem_modules_collection = db['stem_modules']
stem_lessons_collection = db['stem_lessons']
stem_questions_collection = db['stem_questions']
feedback_collection = db['feedback']

# JWT Token decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return {'error': 'Token missing'}, 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            current_user = users_collection.find_one({'_id': data['user_id']})
            if not current_user:
                return {'error': 'Invalid token'}, 401
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired'}, 401
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}, 401
        
        return f(current_user, *args, **kwargs)
    return decorated

# Namespaces
auth_ns = Namespace('auth', description='Authentication operations')
user_ns = Namespace('user', description='User management')
videos_ns = Namespace('videos', description='Video management')
alphabet_ns = Namespace('alphabet', description='Alphabet learning')
vocabulary_ns = Namespace('vocabulary', description='Vocabulary learning')
sentences_ns = Namespace('sentences', description='Sentence learning')
stem_ns = Namespace('stem', description='STEM learning')
convert_ns = Namespace('convert', description='ISL conversion')
progress_ns = Namespace('progress', description='Learning progress')
reports_ns = Namespace('reports', description='Analytics reports')
practice_ns = Namespace('practice', description='Practice sessions')
writing_ns = Namespace('writing', description='Writing practice')
glyphs_ns = Namespace('glyphs', description='Letter glyphs')
health_ns = Namespace('health', description='System health')
feedback_ns = Namespace('feedback', description='User feedback')

api.add_namespace(auth_ns, path='/auth')
api.add_namespace(user_ns, path='/user')
api.add_namespace(videos_ns, path='/videos')
api.add_namespace(alphabet_ns, path='/alphabet')
api.add_namespace(vocabulary_ns, path='/vocabulary')
api.add_namespace(sentences_ns, path='/sentences')
api.add_namespace(stem_ns, path='/stem')
api.add_namespace(convert_ns, path='/convert')
api.add_namespace(progress_ns, path='/progress')
api.add_namespace(reports_ns, path='/reports')
api.add_namespace(practice_ns, path='/practice')
api.add_namespace(writing_ns, path='/writing')
api.add_namespace(glyphs_ns, path='/glyphs')
api.add_namespace(health_ns, path='/health')
api.add_namespace(feedback_ns, path='/feedback')

# Models for Swagger documentation
user_model = api.model('User', {
    'username': fields.String(required=True, description='Username'),
    'email': fields.String(required=True, description='Email address'),
    'password': fields.String(required=True, description='Password'),
    'role': fields.String(description='User role', default='user')
})

login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

video_model = api.model('Video', {
    'title': fields.String(required=True, description='Video title'),
    'url': fields.String(required=True, description='Video URL'),
    'category': fields.String(description='Video category'),
    'duration': fields.Integer(description='Duration in seconds')
})

practice_model = api.model('Practice', {
    'letter': fields.String(required=True, description='Letter to practice'),
    'strokes': fields.List(fields.Raw, required=True, description='Stroke data'),
    'session_id': fields.String(description='Session ID')
})

feedback_model = api.model('Feedback', {
    'type': fields.String(description='Feedback type', enum=['bug', 'feature', 'general', 'improvement']),
    'message': fields.String(required=True, description='Feedback message'),
    'rating': fields.Integer(description='Rating 1-5'),
    'category': fields.String(description='Category', enum=['ui', 'performance', 'content', 'functionality'])
})

text_to_sign_model = api.model('TextToSign', {
    'text': fields.String(required=True, description='Text to convert'),
    'language': fields.String(description='Language code', default='en'),
    'speed': fields.String(description='Playback speed', enum=['slow', 'normal', 'fast'])
})

# Authentication endpoints
@auth_ns.route('/register')
class Register(Resource):
    @auth_ns.expect(user_model)
    @auth_ns.doc('register_user')
    def post(self):
        """Register a new user"""
        data = request.get_json()
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'user')
        
        if not username or not email or not password:
            return {'error': 'Username, email aur password required hai'}, 400
        
        if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
            return {'error': 'Username ya email already exist karta hai'}, 400
        
        password_hash = generate_password_hash(password)
        user = User(username, email, password_hash, role)
        user_dict = user.to_dict()
        
        try:
            result = users_collection.insert_one(user_dict)
            return {
                'message': 'User successfully registered!',
                'user_id': str(result.inserted_id)
            }, 201
        except Exception as e:
            return {'error': f'Database error: {str(e)}'}, 500

@auth_ns.route('/login')
class Login(Resource):
    @auth_ns.expect(login_model)
    @auth_ns.doc('login_user')
    def post(self):
        """Login user and get JWT token"""
        data = request.get_json()
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return {'error': 'Username aur password required hai'}, 400
        
        user = users_collection.find_one({'username': username})
        if not user or not check_password_hash(user['password_hash'], password):
            return {'error': 'Invalid credentials'}, 401
        
        token = jwt.encode({
            'user_id': str(user['_id']),
            'username': user['username'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
        
        return {
            'message': 'Login successful!',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'username': user['username'],
                'email': user['email']
            }
        }, 200

@auth_ns.route('/role')
class Role(Resource):
    @auth_ns.doc('get_user_role', security='Bearer')
    @token_required
    def get(self, current_user):
        """Get current user role"""
        return {
            'role': current_user.get('role', 'user'),
            'username': current_user['username']
        }, 200

# User endpoints
@user_ns.route('/profile')
class Profile(Resource):
    @user_ns.doc('get_user_profile', security='Bearer')
    @token_required
    def get(self, current_user):
        """Get user profile"""
        return {
            'user': {
                'id': str(current_user['_id']),
                'username': current_user['username'],
                'email': current_user['email'],
                'role': current_user.get('role', 'user')
            }
        }, 200

# Health check endpoint
@health_ns.route('')
class Health(Resource):
    @health_ns.doc('health_check')
    def get(self):
        """System health check"""
        try:
            db_status = 'connected'
            try:
                client.admin.command('ping')
            except Exception:
                db_status = 'disconnected'
            
            return {
                'status': 'healthy' if db_status == 'connected' else 'unhealthy',
                'database': {'status': db_status},
                'timestamp': datetime.utcnow().isoformat()
            }, 200 if db_status == 'connected' else 503
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }, 500

# Feedback endpoint
@feedback_ns.route('')
class Feedback(Resource):
    @feedback_ns.expect(feedback_model)
    @feedback_ns.doc('submit_feedback', security='Bearer')
    @token_required
    def post(self, current_user):
        """Submit user feedback"""
        data = request.get_json()
        
        feedback_type = data.get('type', 'general')
        message = data.get('message')
        rating = data.get('rating')
        category = data.get('category', 'general')
        
        if not message:
            return {'error': 'Feedback message required hai'}, 400
        
        try:
            feedback_doc = {
                'user_id': str(current_user['_id']),
                'username': current_user.get('username', 'anonymous'),
                'type': feedback_type,
                'category': category,
                'message': message,
                'rating': rating,
                'status': 'submitted',
                'created_at': datetime.utcnow()
            }
            
            result = feedback_collection.insert_one(feedback_doc)
            
            return {
                'message': 'Feedback successfully submitted!',
                'feedback_id': str(result.inserted_id),
                'status': 'received'
            }, 201
        except Exception as e:
            return {'error': f'Feedback submission error: {str(e)}'}, 500

# Add security definition
api.authorizations = {
    'Bearer': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization',
        'description': 'JWT token. Format: Bearer <token>'
    }
}

if __name__ == '__main__':
    print("Starting Flask app with Swagger documentation...")
    print("Swagger UI available at: http://localhost:5002/docs/")
    print(f"MongoDB URI: {MONGO_URI}")
    app.run(debug=True, port=5002, host='0.0.0.0')