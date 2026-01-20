from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
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

# Swagger API Documentation
api = Api(
    app,
    version='1.0',
    title='ISL Learning Platform API',
    description='Complete API documentation for Indian Sign Language Learning Platform with 21 endpoints',
    doc='/docs/',
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'JWT token. Format: Bearer <your_jwt_token>'
        }
    }
)

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['thadomal_db']

# Collections
users_collection = db['users']
feedback_collection = db['feedback']
videos_collection = db['videos']
practice_sessions_collection = db['practice_sessions']
analytics_events_collection = db['analytics_events']

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
            current_user = users_collection.find_one({'_id': ObjectId(data['user_id'])})
            if not current_user:
                return {'error': 'Invalid token'}, 401
        except jwt.ExpiredSignatureError:
            return {'error': 'Token expired'}, 401
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}, 401
        
        return f(current_user, *args, **kwargs)
    return decorated

# Models for Swagger documentation
user_register_model = api.model('UserRegister', {
    'username': fields.String(required=True, description='Username', example='john_doe'),
    'email': fields.String(required=True, description='Email address', example='john@example.com'),
    'password': fields.String(required=True, description='Password', example='password123'),
    'role': fields.String(description='User role', default='user', example='user')
})

user_login_model = api.model('UserLogin', {
    'username': fields.String(required=True, description='Username', example='john_doe'),
    'password': fields.String(required=True, description='Password', example='password123')
})

feedback_model = api.model('Feedback', {
    'type': fields.String(description='Feedback type', enum=['bug', 'feature', 'general', 'improvement'], example='general'),
    'message': fields.String(required=True, description='Feedback message', example='Great platform!'),
    'rating': fields.Integer(description='Rating 1-5', example=5),
    'category': fields.String(description='Category', enum=['ui', 'performance', 'content', 'functionality'], example='ui')
})

# Additional models
vocabulary_model = api.model('Vocabulary', {
    'letter': fields.String(required=True, description='Starting letter', example='A')
})

sentence_practice_model = api.model('SentencePractice', {
    'sentence_id': fields.String(required=True, description='Sentence ID', example='sample_1'),
    'user_answer': fields.String(required=True, description='User answer', example='I am happy'),
    'correct_answer': fields.String(description='Correct answer', example='I am happy'),
    'practice_type': fields.String(description='Practice type', enum=['translation', 'comprehension', 'formation'], example='translation')
})

writing_practice_model = api.model('WritingPractice', {
    'character': fields.String(required=True, description='Character to practice', example='A'),
    'strokes': fields.List(fields.Raw, required=True, description='Stroke data'),
    'practice_type': fields.String(description='Practice type', enum=['tracing', 'freehand'], example='tracing')
})

sign_to_text_model = api.model('SignToText', {
    'video_data': fields.String(required=True, description='Base64 encoded video data'),
    'confidence_threshold': fields.Float(description='Confidence threshold', example=0.7)
})

speech_to_sign_model = api.model('SpeechToSign', {
    'audio_data': fields.String(required=True, description='Base64 encoded audio data'),
    'language': fields.String(description='Language code', default='en', example='en')
})

text_to_sign_model = api.model('TextToSign', {
    'text': fields.String(required=True, description='Text to convert to ISL', example='Hello how are you'),
    'language': fields.String(description='Language code', default='en', example='en'),
    'speed': fields.String(description='Playback speed', enum=['slow', 'normal', 'fast'], example='normal')
})

# Authentication endpoints
@api.route('/auth/register')
class Register(Resource):
    @api.expect(user_register_model)
    @api.doc('register_user')
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

@api.route('/auth/login')
class Login(Resource):
    @api.expect(user_login_model)
    @api.doc('login_user')
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

@api.route('/auth/role')
class UserRole(Resource):
    @api.doc('get_user_role', security='Bearer')
    @token_required
    def get(self, current_user):
        """Get current user role"""
        return {
            'role': current_user.get('role', 'user'),
            'username': current_user['username']
        }, 200

@api.route('/user/profile')
class UserProfile(Resource):
    @api.doc('get_user_profile', security='Bearer')
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

@api.route('/health')
class HealthCheck(Resource):
    @api.doc('health_check')
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
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            }, 200 if db_status == 'connected' else 503
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }, 500

@api.route('/feedback')
class Feedback(Resource):
    @api.expect(feedback_model)
    @api.doc('submit_feedback', security='Bearer')
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

@api.route('/alphabet/list')
class AlphabetList(Resource):
    @api.doc('get_alphabet_list')
    def get(self):
        """Get English A-Z + numbers 0-9 list"""
        alphabet_list = []
        
        # Add A-Z
        for i in range(26):
            letter = chr(ord('A') + i)
            alphabet_list.append({
                'character': letter,
                'type': 'letter',
                'order': i + 1
            })
        
        # Add 0-9
        for i in range(10):
            alphabet_list.append({
                'character': str(i),
                'type': 'number',
                'order': i + 27
            })
        
        return {
            'alphabet': alphabet_list,
            'total_count': len(alphabet_list),
            'letters_count': 26,
            'numbers_count': 10
        }, 200

@api.route('/alphabet/<string:character>')
class AlphabetCharacter(Resource):
    @api.doc('get_alphabet_character')
    def get(self, character):
        """Get ISL video + tracing data for specific character"""
        character = character.upper()
        if len(character) != 1 or not (character.isalpha() or character.isdigit()):
            return {'error': 'Invalid character - must be single letter or digit'}, 400
        
        return {
            'character': character,
            'type': 'letter' if character.isalpha() else 'number',
            'isl_video_url': f'/videos/isl/{character.lower()}.mp4',
            'tracing_data': {
                'strokes': [],
                'guidelines': {
                    'baseline': 0.8,
                    'midline': 0.5,
                    'topline': 0.2
                }
            },
            'pronunciation': character.lower(),
            'message': 'Sample data - ISL video and tracing data'
        }, 200

@api.route('/convert/text-to-sign')
class TextToSign(Resource):
    @api.expect(text_to_sign_model)
    @api.doc('convert_text_to_sign', security='Bearer')
    @token_required
    def post(self, current_user):
        """Convert English text to ISL output"""
        data = request.get_json()
        
        text = data.get('text')
        language = data.get('language', 'en')
        speed = data.get('speed', 'normal')
        
        if not text:
            return {'error': 'Text required hai'}, 400
        
        words = text.lower().split()
        isl_sequence = []
        
        for i, word in enumerate(words):
            isl_sequence.append({
                'word': word,
                'order': i + 1,
                'isl_video_url': f'/videos/isl/words/{word}.mp4',
                'duration': len(word) * 0.6 if speed == 'normal' else len(word) * 0.8 if speed == 'slow' else len(word) * 0.4,
                'hand_positions': [],
                'facial_expressions': 'neutral'
            })
        
        return {
            'message': 'Text successfully converted to ISL',
            'original_text': text,
            'isl_sequence': isl_sequence,
            'total_duration': sum(sign['duration'] for sign in isl_sequence),
            'word_count': len(words)
        }, 200

@api.route('/vocabulary/<string:letter>')
class VocabularyByLetter(Resource):
    @api.doc('get_vocabulary_by_letter')
    def get(self, letter):
        """Get words starting with specific letter + ISL meanings"""
        letter = letter.upper()
        if len(letter) != 1 or not letter.isalpha():
            return {'error': 'Invalid letter - must be single alphabet character'}, 400
        
        sample_words = {
            'A': ['Apple', 'Ant', 'Air'],
            'B': ['Ball', 'Book', 'Bird'],
            'C': ['Cat', 'Car', 'Cup']
        }
        
        default_words = sample_words.get(letter, [f'{letter}word1', f'{letter}word2'])
        words_data = []
        
        for word in default_words:
            words_data.append({
                'word': word,
                'meaning': f'Meaning of {word}',
                'isl_video_url': f'/videos/vocabulary/{word.lower()}.mp4',
                'difficulty': 'beginner',
                'category': 'general'
            })
        
        return {
            'letter': letter,
            'words': words_data,
            'count': len(words_data)
        }, 200

@api.route('/sentences/<string:level>')
class SentencesByLevel(Resource):
    @api.doc('get_sentences_by_level')
    def get(self, level):
        """Get simple English sentences by difficulty level"""
        valid_levels = ['beginner', 'intermediate', 'advanced']
        if level not in valid_levels:
            return {'error': f'Invalid level - must be one of: {valid_levels}'}, 400
        
        sample_sentences = {
            'beginner': [{'sentence': 'I am happy.', 'words_count': 3}],
            'intermediate': [{'sentence': 'The children are playing.', 'words_count': 4}],
            'advanced': [{'sentence': 'Technology has revolutionized communication.', 'words_count': 4}]
        }
        
        sentences_data = []
        for i, sent in enumerate(sample_sentences.get(level, [])):
            sentences_data.append({
                'id': f'sample_{i+1}',
                'sentence': sent['sentence'],
                'level': level,
                'words_count': sent['words_count'],
                'isl_video_url': f'/videos/sentences/{level}_{i+1}.mp4'
            })
        
        return {
            'level': level,
            'sentences': sentences_data,
            'count': len(sentences_data)
        }, 200

@api.route('/practice/sentence')
class SentencePractice(Resource):
    @api.expect(sentence_practice_model)
    @api.doc('practice_sentence', security='Bearer')
    @token_required
    def post(self, current_user):
        """Submit sentence practice answers"""
        data = request.get_json()
        
        sentence_id = data.get('sentence_id')
        user_answer = data.get('user_answer')
        
        if not sentence_id or not user_answer:
            return {'error': 'Sentence ID aur user answer required hai'}, 400
        
        score = 85  # Mock score
        
        return {
            'message': 'Sentence practice submitted successfully!',
            'score': score,
            'feedback': 'Good job!'
        }, 201

@api.route('/writing/practice')
class WritingPractice(Resource):
    @api.expect(writing_practice_model)
    @api.doc('writing_practice', security='Bearer')
    @token_required
    def post(self, current_user):
        """Submit writing pad practice"""
        data = request.get_json()
        
        character = data.get('character')
        strokes = data.get('strokes', [])
        
        if not character or not strokes:
            return {'error': 'Character aur strokes required hai'}, 400
        
        score = min(100, len(strokes) * 8)
        
        return {
            'message': 'Writing practice submitted successfully!',
            'score': score
        }, 201

@api.route('/stem/modules')
class STEMModules(Resource):
    @api.doc('get_stem_modules')
    def get(self):
        """Get STEM topics list (Math + Science)"""
        sample_modules = [
            {
                'id': 'math_basic',
                'title': 'Basic Mathematics',
                'subject': 'math',
                'topics': ['Addition', 'Subtraction'],
                'lessons_count': 12
            }
        ]
        
        return {
            'modules': sample_modules,
            'count': len(sample_modules)
        }, 200

@api.route('/stem/lesson/<string:lesson_id>')
class STEMLesson(Resource):
    @api.doc('get_stem_lesson')
    def get(self, lesson_id):
        """Get lesson ISL video + content"""
        return {
            'lesson': {
                'lesson_id': lesson_id,
                'title': f'Lesson {lesson_id}',
                'isl_video_url': f'/videos/stem/{lesson_id}.mp4',
                'content': 'Sample lesson content'
            }
        }, 200

@api.route('/stem/questions/<string:lesson_id>')
class STEMQuestions(Resource):
    @api.doc('get_stem_questions')
    def get(self, lesson_id):
        """Get auto-generated questions"""
        return {
            'lesson_id': lesson_id,
            'questions': [
                {
                    'id': f'{lesson_id}_q1',
                    'question': 'What is 5 + 3?',
                    'correct_answer': '8'
                }
            ],
            'count': 1
        }, 200

@api.route('/convert/sign-to-text')
class SignToText(Resource):
    @api.expect(sign_to_text_model)
    @api.doc('convert_sign_to_text', security='Bearer')
    @token_required
    def post(self, current_user):
        """Convert sign to text"""
        return {
            'recognized_text': 'hello world',
            'confidence_score': 0.85
        }, 200

@api.route('/convert/speech-to-sign')
class SpeechToSign(Resource):
    @api.expect(speech_to_sign_model)
    @api.doc('convert_speech_to_sign', security='Bearer')
    @token_required
    def post(self, current_user):
        """Convert speech to ISL"""
        return {
            'transcribed_text': 'hello world',
            'isl_sequence': [{'word': 'hello', 'isl_video_url': '/videos/hello.mp4'}]
        }, 200

@api.route('/progress/overview')
class ProgressOverview(Resource):
    @api.doc('get_progress_overview', security='Bearer')
    @token_required
    def get(self, current_user):
        """Get overall performance"""
        return {
            'overall_performance': {
                'total_sessions': 25,
                'average_score': 85.5
            }
        }, 200

@api.route('/progress/lesson/<string:lesson_id>')
class ProgressLesson(Resource):
    @api.doc('get_lesson_progress', security='Bearer')
    @token_required
    def get(self, current_user, lesson_id):
        """Get lesson-wise progress"""
        return {
            'lesson_id': lesson_id,
            'progress': {
                'completion_percentage': 75.0,
                'best_score': 95
            }
        }, 200

@api.route('/reports/visual')
class VisualReports(Resource):
    @api.doc('get_visual_reports', security='Bearer')
    @token_required
    def get(self, current_user):
        """Get graph/chart data"""
        return {
            'charts': {
                'performance_chart': {
                    'data': [{'session': 1, 'score': 85}]
                }
            }
        }, 200

if __name__ == '__main__':
    print("Starting Flask app with Swagger documentation...")
    print("Swagger UI available at: http://localhost:5002/docs/")
    print("API Base URL: http://localhost:5002/")
    print(f"MongoDB URI: {MONGO_URI.split('@')[1] if '@' in MONGO_URI else 'local'}")
    print("\nAPI Endpoints in Swagger:")
    print("Authentication: /auth/register, /auth/login, /auth/role")
    print("User: /user/profile")
    print("Alphabet: /alphabet/list, /alphabet/<character>")
    print("Vocabulary: /vocabulary/<letter>")
    print("Sentences: /sentences/<level>")
    print("Practice: /practice/sentence, /writing/practice")
    print("STEM: /stem/modules, /stem/lesson/<id>, /stem/questions/<id>")
    print("Convert: /convert/text-to-sign, /convert/sign-to-text, /convert/speech-to-sign")
    print("Progress: /progress/overview, /progress/lesson/<id>")
    print("Reports: /reports/visual")
    print("System: /health, /feedback")
    print("\nTotal: 21 APIs in Swagger UI!")
    print("\nSwagger UI Ready!")
    print("\n" + "="*50)
    app.run(debug=True, port=5002, host='0.0.0.0')