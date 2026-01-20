from flask import Flask, request, jsonify
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
            return jsonify({'error': 'Token missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            
            # Convert user_id to ObjectId for MongoDB query
            try:
                user_id = ObjectId(data['user_id'])
            except:
                user_id = data['user_id']
            
            current_user = users_collection.find_one({'_id': user_id})
            if not current_user:
                return jsonify({'error': 'Invalid token - user not found'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            return jsonify({'error': f'Token validation error: {str(e)}'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/')
def home():
    return jsonify({'message': 'Flask backend is running!'})

@app.route('/test', methods=['GET', 'POST'])
def test_route():
    if request.method == 'GET':
        return jsonify({'message': 'Test route working - GET method'})
    else:
        return jsonify({'message': 'Test route working - POST method', 'data': request.get_json()})

@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not email or not password:
        return jsonify({'error': 'Username, email aur password required hai'}), 400
    
    # Check if user already exists
    if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
        return jsonify({'error': 'Username ya email already exist karta hai'}), 400
    
    password_hash = generate_password_hash(password)
    
    # Create user using model
    user = User(username, email, password_hash, role)
    user_dict = user.to_dict()
    
    try:
        result = users_collection.insert_one(user_dict)
        return jsonify({
            'message': 'User successfully registered!',
            'user_id': str(result.inserted_id)
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username aur password required hai'}), 400
    
    # Find user
    user = users_collection.find_one({'username': username})
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Generate JWT token
    token = jwt.encode({
        'user_id': str(user['_id']),
        'username': user['username'],
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
    
    return jsonify({
        'message': 'Login successful!',
        'token': token,
        'user': {
            'id': str(user['_id']),
            'username': user['username'],
            'email': user['email']
        }
    }), 200

@app.route('/auth/role', methods=['GET'])
@token_required
def get_role(current_user):
    return jsonify({
        'role': current_user.get('role', 'user'),
        'username': current_user['username']
    }), 200

@app.route('/user/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    return jsonify({
        'user': {
            'id': str(current_user['_id']),
            'username': current_user['username'],
            'email': current_user['email'],
            'role': current_user.get('role', 'user')
        }
    }), 200

# Video endpoints
@app.route('/videos', methods=['GET'])
def get_videos():
    category = request.args.get('category')
    
    # Build query
    query = {'is_public': True}
    if category:
        query['category'] = category
    
    try:
        videos = list(videos_collection.find(query))
        
        # Convert ObjectId to string
        for video in videos:
            video['_id'] = str(video['_id'])
            if video.get('user_id'):
                video['user_id'] = str(video['user_id'])
        
        return jsonify({
            'videos': videos,
            'count': len(videos)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/videos/<video_id>', methods=['GET'])
def get_video(video_id):
    try:
        # Validate ObjectId format
        if len(video_id) != 24:
            return jsonify({'error': 'Invalid video ID format - must be 24 characters'}), 400
            
        if not ObjectId.is_valid(video_id):
            return jsonify({'error': 'Invalid video ID format'}), 400
        
        video = videos_collection.find_one({'_id': ObjectId(video_id)})
        
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Convert ObjectId to string
        video['_id'] = str(video['_id'])
        if video.get('user_id'):
            video['user_id'] = str(video['user_id'])
        
        return jsonify({'video': video}), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/videos', methods=['POST'])
@token_required
def create_video(current_user):
    data = request.get_json()
    
    title = data.get('title')
    url = data.get('url')
    category = data.get('category')
    duration = data.get('duration', 0)
    
    if not title or not url:
        return jsonify({'error': 'Title aur URL required hai'}), 400
    
    # Create video using model
    video = Video(title, url, duration, str(current_user['_id']), category)
    video_dict = video.to_dict()
    
    try:
        result = videos_collection.insert_one(video_dict)
        return jsonify({
            'message': 'Video successfully created!',
            'video_id': str(result.inserted_id)
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

# Writing pad endpoints
@app.route('/glyphs/<letter>', methods=['GET'])
def get_glyph(letter):
    try:
        # Convert to uppercase for consistency
        letter = letter.upper()
        
        # Validate single letter
        if len(letter) != 1 or not letter.isalpha():
            return jsonify({'error': 'Invalid letter - must be single alphabet character'}), 400
        
        glyph = glyphs_collection.find_one({'letter': letter})
        
        if not glyph:
            # Return default glyph structure if not found
            return jsonify({
                'letter': letter,
                'strokes': [],
                'guidelines': {
                    'baseline': 0.8,
                    'midline': 0.5,
                    'topline': 0.2
                },
                'message': 'Glyph data not found - using default'
            }), 200
        
        # Convert ObjectId to string
        glyph['_id'] = str(glyph['_id'])
        
        return jsonify({'glyph': glyph}), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/practice/submit', methods=['POST'])
@token_required
def submit_practice(current_user):
    data = request.get_json()
    
    letter = data.get('letter')
    strokes = data.get('strokes', [])
    session_id = data.get('session_id')
    
    if not letter or not strokes:
        return jsonify({'error': 'Letter aur strokes required hai'}), 400
    
    try:
        # Calculate basic score (placeholder logic)
        score = min(100, len(strokes) * 10)  # Simple scoring
        
        # Create practice session
        practice_session = PracticeSession(
            user_id=str(current_user['_id']),
            video_id=None,  # For writing practice
            session_type='writing_practice'
        )
        practice_session.score = score
        practice_session.completed = True
        practice_session.notes = f"Letter: {letter}, Strokes: {len(strokes)}"
        
        session_dict = practice_session.to_dict()
        session_dict['letter'] = letter
        session_dict['strokes'] = strokes
        
        result = practice_sessions_collection.insert_one(session_dict)
        
        # Log analytics event
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='practice_submit',
            event_data={
                'letter': letter,
                'score': score,
                'stroke_count': len(strokes)
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Practice submitted successfully!',
            'session_id': str(result.inserted_id),
            'score': score,
            'feedback': 'Good job!' if score > 50 else 'Keep practicing!'
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/practice/score', methods=['GET'])
@token_required
def get_practice_scores(current_user):
    try:
        # Get recent practice sessions
        sessions = list(practice_sessions_collection.find({
            'user_id': str(current_user['_id']),
            'session_type': 'writing_practice'
        }).sort('created_at', -1).limit(10))
        
        # Convert ObjectId to string
        for session in sessions:
            session['_id'] = str(session['_id'])
        
        # Calculate average score
        total_score = sum(session.get('score', 0) for session in sessions)
        avg_score = total_score / len(sessions) if sessions else 0
        
        return jsonify({
            'sessions': sessions,
            'average_score': round(avg_score, 2),
            'total_sessions': len(sessions)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

# Alphabet & Numeracy Learning endpoints
@app.route('/alphabet/list', methods=['GET'])
def get_alphabet_list():
    try:
        # Generate A-Z and 0-9 list
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
        
        return jsonify({
            'alphabet': alphabet_list,
            'total_count': len(alphabet_list),
            'letters_count': 26,
            'numbers_count': 10
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/alphabet/<character>', methods=['GET'])
def get_alphabet_character(character):
    try:
        # Validate character
        character = character.upper()
        if len(character) != 1 or not (character.isalpha() or character.isdigit()):
            return jsonify({'error': 'Invalid character - must be single letter or digit'}), 400
        
        # Get ISL video and tracing data
        alphabet_data = alphabet_collection.find_one({'character': character})
        
        if not alphabet_data:
            # Return default structure if not found
            return jsonify({
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
                'message': 'Default data - ISL video and tracing data not found'
            }), 200
        
        # Convert ObjectId to string
        alphabet_data['_id'] = str(alphabet_data['_id'])
        
        return jsonify({
            'character_data': alphabet_data
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/writing/practice', methods=['POST'])
@token_required
def writing_practice(current_user):
    data = request.get_json()
    
    character = data.get('character')
    strokes = data.get('strokes', [])
    practice_type = data.get('practice_type', 'tracing')  # tracing, freehand
    
    if not character or not strokes:
        return jsonify({'error': 'Character aur strokes required hai'}), 400
    
    try:
        # Calculate score based on stroke accuracy
        score = min(100, len(strokes) * 8)  # Simple scoring for writing
        
        # Create practice session
        practice_session = PracticeSession(
            user_id=str(current_user['_id']),
            video_id=None,
            session_type='writing_practice'
        )
        practice_session.score = score
        practice_session.completed = True
        practice_session.notes = f"Character: {character}, Type: {practice_type}, Strokes: {len(strokes)}"
        
        session_dict = practice_session.to_dict()
        session_dict['character'] = character
        session_dict['strokes'] = strokes
        session_dict['practice_type'] = practice_type
        
        result = practice_sessions_collection.insert_one(session_dict)
        
        # Log analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='writing_practice',
            event_data={
                'character': character,
                'practice_type': practice_type,
                'score': score,
                'stroke_count': len(strokes)
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Writing practice submitted successfully!',
            'session_id': str(result.inserted_id),
            'score': score,
            'feedback': 'Excellent!' if score > 80 else 'Good job!' if score > 50 else 'Keep practicing!'
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

# Vocabulary & Sentence Builder endpoints
@app.route('/vocabulary/<letter>', methods=['GET'])
def get_vocabulary_by_letter(letter):
    try:
        # Validate letter
        letter = letter.upper()
        if len(letter) != 1 or not letter.isalpha():
            return jsonify({'error': 'Invalid letter - must be single alphabet character'}), 400
        
        # Get words starting with the letter
        words = list(vocabulary_collection.find({'starting_letter': letter}))
        
        if not words:
            # Return sample words if not found in database
            sample_words = {
                'A': ['Apple', 'Ant', 'Air'],
                'B': ['Ball', 'Book', 'Bird'],
                'C': ['Cat', 'Car', 'Cup'],
                'D': ['Dog', 'Door', 'Duck'],
                'E': ['Egg', 'Eye', 'Ear']
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
            
            return jsonify({
                'letter': letter,
                'words': words_data,
                'count': len(words_data),
                'message': 'Sample data - vocabulary not found in database'
            }), 200
        
        # Convert ObjectId to string
        for word in words:
            word['_id'] = str(word['_id'])
        
        return jsonify({
            'letter': letter,
            'words': words,
            'count': len(words)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/sentences/<level>', methods=['GET'])
def get_sentences_by_level(level):
    try:
        # Validate level
        valid_levels = ['beginner', 'intermediate', 'advanced']
        if level not in valid_levels:
            return jsonify({'error': f'Invalid level - must be one of: {valid_levels}'}), 400
        
        # Get sentences by level
        sentences = list(sentences_collection.find({'level': level}).limit(10))
        
        if not sentences:
            # Return sample sentences if not found
            sample_sentences = {
                'beginner': [
                    {'sentence': 'I am happy.', 'words_count': 3},
                    {'sentence': 'The cat is big.', 'words_count': 4},
                    {'sentence': 'She likes books.', 'words_count': 3}
                ],
                'intermediate': [
                    {'sentence': 'The children are playing in the park.', 'words_count': 7},
                    {'sentence': 'My mother cooks delicious food.', 'words_count': 5},
                    {'sentence': 'We went to the market yesterday.', 'words_count': 6}
                ],
                'advanced': [
                    {'sentence': 'The weather forecast predicts heavy rainfall tomorrow.', 'words_count': 8},
                    {'sentence': 'Technology has revolutionized modern communication methods.', 'words_count': 7},
                    {'sentence': 'Environmental conservation requires collective global efforts.', 'words_count': 7}
                ]
            }
            
            default_sentences = sample_sentences.get(level, [])
            sentences_data = []
            
            for i, sent in enumerate(default_sentences):
                sentences_data.append({
                    'id': f'sample_{i+1}',
                    'sentence': sent['sentence'],
                    'level': level,
                    'words_count': sent['words_count'],
                    'isl_video_url': f'/videos/sentences/{level}_{i+1}.mp4',
                    'difficulty_score': {'beginner': 1, 'intermediate': 5, 'advanced': 8}[level]
                })
            
            return jsonify({
                'level': level,
                'sentences': sentences_data,
                'count': len(sentences_data),
                'message': 'Sample data - sentences not found in database'
            }), 200
        
        # Convert ObjectId to string
        for sentence in sentences:
            sentence['_id'] = str(sentence['_id'])
        
        return jsonify({
            'level': level,
            'sentences': sentences,
            'count': len(sentences)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/practice/sentence', methods=['POST'])
@token_required
def practice_sentence(current_user):
    data = request.get_json()
    
    sentence_id = data.get('sentence_id')
    user_answer = data.get('user_answer')
    correct_answer = data.get('correct_answer')
    practice_type = data.get('practice_type', 'translation')  # translation, comprehension, formation
    
    if not sentence_id or not user_answer:
        return jsonify({'error': 'Sentence ID aur user answer required hai'}), 400
    
    try:
        # Calculate score based on answer accuracy
        if correct_answer:
            # Simple string similarity check
            user_words = set(user_answer.lower().split())
            correct_words = set(correct_answer.lower().split())
            
            if user_words == correct_words:
                score = 100
            else:
                # Calculate partial score
                common_words = user_words.intersection(correct_words)
                score = int((len(common_words) / len(correct_words)) * 100) if correct_words else 0
        else:
            score = 75  # Default score when no correct answer provided
        
        # Create practice session
        practice_session = PracticeSession(
            user_id=str(current_user['_id']),
            video_id=None,
            session_type='sentence_practice'
        )
        practice_session.score = score
        practice_session.completed = True
        practice_session.notes = f"Sentence ID: {sentence_id}, Type: {practice_type}"
        
        session_dict = practice_session.to_dict()
        session_dict['sentence_id'] = sentence_id
        session_dict['user_answer'] = user_answer
        session_dict['correct_answer'] = correct_answer
        session_dict['practice_type'] = practice_type
        
        result = practice_sessions_collection.insert_one(session_dict)
        
        # Log analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='sentence_practice',
            event_data={
                'sentence_id': sentence_id,
                'practice_type': practice_type,
                'score': score,
                'answer_length': len(user_answer.split())
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Sentence practice submitted successfully!',
            'session_id': str(result.inserted_id),
            'score': score,
            'feedback': 'Perfect!' if score == 100 else 'Good job!' if score > 70 else 'Keep practicing!',
            'is_correct': score == 100
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

# AI Bridge endpoints
@app.route('/ai/text-to-sign', methods=['POST'])
@token_required
def text_to_sign(current_user):
    data = request.get_json()
    
    text = data.get('text')
    language = data.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'Text required hai'}), 400
    
    try:
        # Placeholder AI processing - replace with actual AI service
        sign_sequence = []
        words = text.lower().split()
        
        for word in words:
            sign_sequence.append({
                'word': word,
                'sign_video_url': f'/videos/signs/{word}.mp4',
                'duration': len(word) * 0.5,  # Placeholder duration
                'hand_positions': []  # Placeholder for hand position data
            })
        
        # Log analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='text_to_sign',
            event_data={
                'text': text,
                'language': language,
                'word_count': len(words)
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Text converted to sign language',
            'original_text': text,
            'sign_sequence': sign_sequence,
            'total_duration': sum(sign['duration'] for sign in sign_sequence)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'AI processing error: {str(e)}'}), 500

@app.route('/ai/sign-to-text', methods=['POST'])
@token_required
def sign_to_text(current_user):
    data = request.get_json()
    
    video_data = data.get('video_data')  # Base64 or video file path
    hand_positions = data.get('hand_positions', [])
    
    if not video_data and not hand_positions:
        return jsonify({'error': 'Video data ya hand positions required hai'}), 400
    
    try:
        # Placeholder AI processing - replace with actual AI service
        recognized_words = ['hello', 'world']  # Placeholder recognition
        confidence_score = 0.85
        
        recognized_text = ' '.join(recognized_words)
        
        # Log analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='sign_to_text',
            event_data={
                'recognized_text': recognized_text,
                'confidence': confidence_score,
                'word_count': len(recognized_words)
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Sign language converted to text',
            'recognized_text': recognized_text,
            'confidence_score': confidence_score,
            'recognized_words': recognized_words
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'AI processing error: {str(e)}'}), 500

@app.route('/ai/speech-to-sign', methods=['POST'])
@token_required
def speech_to_sign(current_user):
    data = request.get_json()
    
    audio_data = data.get('audio_data')  # Base64 or audio file path
    language = data.get('language', 'en')
    
    if not audio_data:
        return jsonify({'error': 'Audio data required hai'}), 400
    
    try:
        # Placeholder AI processing - replace with actual AI service
        # Step 1: Speech to text
        transcribed_text = "hello how are you"  # Placeholder transcription
        
        # Step 2: Text to sign
        words = transcribed_text.split()
        sign_sequence = []
        
        for word in words:
            sign_sequence.append({
                'word': word,
                'sign_video_url': f'/videos/signs/{word}.mp4',
                'duration': len(word) * 0.5,
                'hand_positions': []
            })
        
        # Log analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='speech_to_sign',
            event_data={
                'transcribed_text': transcribed_text,
                'language': language,
                'word_count': len(words)
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Speech converted to sign language',
            'transcribed_text': transcribed_text,
            'sign_sequence': sign_sequence,
            'total_duration': sum(sign['duration'] for sign in sign_sequence)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'AI processing error: {str(e)}'}), 500

# STEM Learning endpoints
@app.route('/stem/modules', methods=['GET'])
def get_stem_modules():
    try:
        # Get STEM modules from database
        modules = list(stem_modules_collection.find())
        
        if not modules:
            # Return sample STEM modules if not found
            sample_modules = [
                {
                    'id': 'math_basic',
                    'title': 'Basic Mathematics',
                    'subject': 'math',
                    'level': 'beginner',
                    'topics': ['Addition', 'Subtraction', 'Multiplication', 'Division'],
                    'lessons_count': 12
                },
                {
                    'id': 'science_plants',
                    'title': 'Plant Biology',
                    'subject': 'science',
                    'level': 'intermediate',
                    'topics': ['Photosynthesis', 'Plant Parts', 'Growth'],
                    'lessons_count': 8
                },
                {
                    'id': 'math_geometry',
                    'title': 'Basic Geometry',
                    'subject': 'math',
                    'level': 'intermediate',
                    'topics': ['Shapes', 'Angles', 'Area', 'Perimeter'],
                    'lessons_count': 15
                }
            ]
            
            return jsonify({
                'modules': sample_modules,
                'count': len(sample_modules),
                'subjects': ['math', 'science'],
                'message': 'Sample data - STEM modules not found in database'
            }), 200
        
        # Convert ObjectId to string
        for module in modules:
            module['_id'] = str(module['_id'])
        
        return jsonify({
            'modules': modules,
            'count': len(modules)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/stem/lesson/<lesson_id>', methods=['GET'])
def get_stem_lesson(lesson_id):
    try:
        # Validate lesson_id
        if len(lesson_id) == 24 and ObjectId.is_valid(lesson_id):
            lesson = stem_lessons_collection.find_one({'_id': ObjectId(lesson_id)})
        else:
            lesson = stem_lessons_collection.find_one({'lesson_id': lesson_id})
        
        if not lesson:
            # Return sample lesson if not found
            sample_lesson = {
                'lesson_id': lesson_id,
                'title': f'Sample Lesson {lesson_id}',
                'subject': 'math',
                'module_id': 'math_basic',
                'content': {
                    'text': 'This is a sample lesson content about basic mathematics.',
                    'examples': ['2 + 2 = 4', '5 - 3 = 2'],
                    'key_points': ['Addition combines numbers', 'Subtraction finds difference']
                },
                'isl_video_url': f'/videos/stem/lessons/{lesson_id}.mp4',
                'duration': 300,  # 5 minutes
                'difficulty': 'beginner',
                'prerequisites': [],
                'message': 'Sample data - lesson not found in database'
            }
            
            return jsonify({
                'lesson': sample_lesson
            }), 200
        
        # Convert ObjectId to string
        lesson['_id'] = str(lesson['_id'])
        
        return jsonify({
            'lesson': lesson
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/stem/questions/<lesson_id>', methods=['GET'])
def get_stem_questions(lesson_id):
    try:
        # Get questions for the lesson
        questions = list(stem_questions_collection.find({'lesson_id': lesson_id}))
        
        if not questions:
            # Generate sample auto-generated questions
            sample_questions = [
                {
                    'id': f'{lesson_id}_q1',
                    'lesson_id': lesson_id,
                    'question': 'What is 5 + 3?',
                    'type': 'multiple_choice',
                    'options': ['6', '7', '8', '9'],
                    'correct_answer': '8',
                    'difficulty': 'easy',
                    'points': 10
                },
                {
                    'id': f'{lesson_id}_q2',
                    'lesson_id': lesson_id,
                    'question': 'Solve: 12 - 4 = ?',
                    'type': 'fill_blank',
                    'correct_answer': '8',
                    'difficulty': 'easy',
                    'points': 10
                },
                {
                    'id': f'{lesson_id}_q3',
                    'lesson_id': lesson_id,
                    'question': 'Which operation is used to combine numbers?',
                    'type': 'multiple_choice',
                    'options': ['Addition', 'Subtraction', 'Division', 'None'],
                    'correct_answer': 'Addition',
                    'difficulty': 'medium',
                    'points': 15
                }
            ]
            
            return jsonify({
                'lesson_id': lesson_id,
                'questions': sample_questions,
                'count': len(sample_questions),
                'total_points': sum(q['points'] for q in sample_questions),
                'message': 'Auto-generated sample questions - not found in database'
            }), 200
        
        # Convert ObjectId to string
        for question in questions:
            question['_id'] = str(question['_id'])
        
        return jsonify({
            'lesson_id': lesson_id,
            'questions': questions,
            'count': len(questions),
            'total_points': sum(q.get('points', 0) for q in questions)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

# English ↔ ISL Converter (CORE) endpoints
@app.route('/convert/text-to-sign', methods=['POST'])
@token_required
def convert_text_to_sign(current_user):
    data = request.get_json()
    
    text = data.get('text')
    language = data.get('language', 'en')
    speed = data.get('speed', 'normal')  # slow, normal, fast
    
    if not text:
        return jsonify({'error': 'Text required hai'}), 400
    
    try:
        # Enhanced ISL conversion processing
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
        
        # Log conversion analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='text_to_isl_conversion',
            event_data={
                'original_text': text,
                'word_count': len(words),
                'language': language,
                'speed': speed
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Text successfully converted to ISL',
            'original_text': text,
            'isl_sequence': isl_sequence,
            'total_duration': sum(sign['duration'] for sign in isl_sequence),
            'word_count': len(words),
            'conversion_id': str(analytics_event.to_dict().get('_id', 'temp_id'))
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Conversion error: {str(e)}'}), 500

@app.route('/convert/sign-to-text', methods=['POST'])
@token_required
def convert_sign_to_text(current_user):
    data = request.get_json()
    
    video_data = data.get('video_data')  # Base64 encoded video
    camera_input = data.get('camera_input', True)
    confidence_threshold = data.get('confidence_threshold', 0.7)
    
    if not video_data:
        return jsonify({'error': 'Video data required hai for sign recognition'}), 400
    
    try:
        # Enhanced ISL to text processing
        # Placeholder for actual AI/ML processing
        recognized_signs = ['hello', 'how', 'are', 'you']  # Mock recognition
        confidence_scores = [0.95, 0.88, 0.92, 0.85]  # Mock confidence
        
        # Filter by confidence threshold
        filtered_results = []
        for sign, confidence in zip(recognized_signs, confidence_scores):
            if confidence >= confidence_threshold:
                filtered_results.append({
                    'sign': sign,
                    'confidence': confidence,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        recognized_text = ' '.join([result['sign'] for result in filtered_results])
        avg_confidence = sum([result['confidence'] for result in filtered_results]) / len(filtered_results) if filtered_results else 0
        
        # Log recognition analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='isl_to_text_conversion',
            event_data={
                'recognized_text': recognized_text,
                'confidence_threshold': confidence_threshold,
                'signs_detected': len(recognized_signs),
                'signs_accepted': len(filtered_results),
                'avg_confidence': avg_confidence
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'ISL signs successfully converted to text',
            'recognized_text': recognized_text,
            'detailed_results': filtered_results,
            'average_confidence': round(avg_confidence, 2),
            'signs_detected': len(recognized_signs),
            'signs_accepted': len(filtered_results),
            'conversion_id': str(analytics_event.to_dict().get('_id', 'temp_id'))
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Recognition error: {str(e)}'}), 500

@app.route('/convert/speech-to-sign', methods=['POST'])
@token_required
def convert_speech_to_sign(current_user):
    data = request.get_json()
    
    audio_data = data.get('audio_data')  # Base64 encoded audio
    language = data.get('language', 'en')
    speech_speed = data.get('speech_speed', 'normal')
    
    if not audio_data:
        return jsonify({'error': 'Audio data required hai for speech recognition'}), 400
    
    try:
        # Step 1: Speech to Text (Enhanced)
        transcribed_text = "hello how are you today"  # Mock transcription
        transcription_confidence = 0.92
        
        # Step 2: Text to ISL conversion
        words = transcribed_text.split()
        isl_sequence = []
        
        for i, word in enumerate(words):
            isl_sequence.append({
                'word': word,
                'order': i + 1,
                'isl_video_url': f'/videos/isl/words/{word}.mp4',
                'duration': len(word) * 0.5,
                'hand_positions': [],
                'facial_expressions': 'neutral',
                'difficulty': 'beginner'
            })
        
        # Log speech-to-ISL analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='speech_to_isl_conversion',
            event_data={
                'transcribed_text': transcribed_text,
                'transcription_confidence': transcription_confidence,
                'language': language,
                'word_count': len(words),
                'speech_speed': speech_speed
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Speech successfully converted to ISL',
            'transcribed_text': transcribed_text,
            'transcription_confidence': transcription_confidence,
            'isl_sequence': isl_sequence,
            'total_duration': sum(sign['duration'] for sign in isl_sequence),
            'word_count': len(words),
            'conversion_id': str(analytics_event.to_dict().get('_id', 'temp_id'))
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Speech conversion error: {str(e)}'}), 500

# Learning Analytics Dashboard endpoints
@app.route('/progress/overview', methods=['GET'])
@token_required
def get_progress_overview(current_user):
    try:
        user_id = str(current_user['_id'])
        
        # Get all practice sessions
        sessions = list(practice_sessions_collection.find({'user_id': user_id}))
        
        # Get analytics events
        events = list(analytics_events_collection.find({'user_id': user_id}).sort('created_at', -1).limit(50))
        
        # Calculate overall statistics
        total_sessions = len(sessions)
        completed_sessions = len([s for s in sessions if s.get('completed', False)])
        avg_score = sum(s.get('score', 0) for s in sessions) / total_sessions if total_sessions > 0 else 0
        
        # Activity by type
        activity_types = {}
        for session in sessions:
            session_type = session.get('session_type', 'unknown')
            activity_types[session_type] = activity_types.get(session_type, 0) + 1
        
        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_sessions = [s for s in sessions if s.get('created_at', datetime.min) > week_ago]
        
        # Learning streaks
        learning_days = set()
        for session in sessions:
            if session.get('created_at'):
                learning_days.add(session['created_at'].date() if hasattr(session['created_at'], 'date') else datetime.utcnow().date())
        
        return jsonify({
            'user_id': user_id,
            'overall_performance': {
                'total_sessions': total_sessions,
                'completed_sessions': completed_sessions,
                'completion_rate': round((completed_sessions / total_sessions * 100) if total_sessions > 0 else 0, 2),
                'average_score': round(avg_score, 2),
                'learning_streak_days': len(learning_days)
            },
            'activity_breakdown': activity_types,
            'recent_activity': {
                'last_7_days': len(recent_sessions),
                'sessions_this_week': recent_sessions[:5]  # Latest 5 sessions
            },
            'skill_areas': {
                'writing_practice': len([s for s in sessions if 'writing' in s.get('session_type', '')]),
                'vocabulary': len([e for e in events if 'vocabulary' in e.get('event_type', '')]),
                'stem_learning': len([e for e in events if 'stem' in e.get('event_type', '')]),
                'isl_conversion': len([e for e in events if 'isl' in e.get('event_type', '')])
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Analytics error: {str(e)}'}), 500

@app.route('/progress/lesson/<lesson_id>', methods=['GET'])
@token_required
def get_lesson_progress(current_user, lesson_id):
    try:
        user_id = str(current_user['_id'])
        
        # Get lesson-specific sessions
        lesson_sessions = list(practice_sessions_collection.find({
            'user_id': user_id,
            '$or': [
                {'lesson_id': lesson_id},
                {'video_id': lesson_id},
                {'notes': {'$regex': lesson_id, '$options': 'i'}}
            ]
        }).sort('created_at', -1))
        
        # Get lesson-specific analytics events
        lesson_events = list(analytics_events_collection.find({
            'user_id': user_id,
            'event_data.lesson_id': lesson_id
        }).sort('created_at', -1))
        
        if not lesson_sessions and not lesson_events:
            # Return sample progress data
            return jsonify({
                'lesson_id': lesson_id,
                'progress': {
                    'completion_percentage': 0,
                    'attempts': 0,
                    'best_score': 0,
                    'average_score': 0,
                    'time_spent': 0,
                    'status': 'not_started'
                },
                'attempts_history': [],
                'learning_path': {
                    'current_step': 1,
                    'total_steps': 5,
                    'next_recommended': f'{lesson_id}_next'
                },
                'message': 'No progress data found - lesson not started'
            }), 200
        
        # Calculate lesson progress
        total_attempts = len(lesson_sessions)
        completed_attempts = len([s for s in lesson_sessions if s.get('completed', False)])
        scores = [s.get('score', 0) for s in lesson_sessions if s.get('score') is not None]
        
        best_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Calculate time spent
        total_time = sum(s.get('duration', 0) for s in lesson_sessions)
        
        # Determine completion percentage
        completion_percentage = min(100, (completed_attempts / max(1, total_attempts)) * 100)
        
        return jsonify({
            'lesson_id': lesson_id,
            'progress': {
                'completion_percentage': round(completion_percentage, 2),
                'attempts': total_attempts,
                'completed_attempts': completed_attempts,
                'best_score': best_score,
                'average_score': round(avg_score, 2),
                'time_spent_minutes': round(total_time / 60, 2),
                'status': 'completed' if completion_percentage >= 100 else 'in_progress' if completion_percentage > 0 else 'not_started'
            },
            'attempts_history': [
                {
                    'attempt_number': i + 1,
                    'score': session.get('score', 0),
                    'completed': session.get('completed', False),
                    'date': session.get('created_at', datetime.utcnow()).isoformat() if hasattr(session.get('created_at'), 'isoformat') else str(session.get('created_at', 'unknown'))
                }
                for i, session in enumerate(lesson_sessions[:10])  # Last 10 attempts
            ],
            'learning_path': {
                'current_step': min(5, max(1, int(completion_percentage / 20))),
                'total_steps': 5,
                'next_recommended': f'{lesson_id}_advanced' if completion_percentage >= 80 else f'{lesson_id}_practice'
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Progress tracking error: {str(e)}'}), 500

@app.route('/reports/visual', methods=['GET'])
@token_required
def get_visual_reports(current_user):
    try:
        user_id = str(current_user['_id'])
        report_type = request.args.get('type', 'all')  # all, performance, activity, skills
        
        # Get user data
        sessions = list(practice_sessions_collection.find({'user_id': user_id}).sort('created_at', 1))
        events = list(analytics_events_collection.find({'user_id': user_id}).sort('created_at', 1))
        
        # Performance chart data (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_sessions = [s for s in sessions if s.get('created_at', datetime.min) > thirty_days_ago]
        
        performance_data = []
        for i, session in enumerate(recent_sessions[-20:]):  # Last 20 sessions
            performance_data.append({
                'session': i + 1,
                'score': session.get('score', 0),
                'date': session.get('created_at', datetime.utcnow()).strftime('%Y-%m-%d') if hasattr(session.get('created_at'), 'strftime') else 'unknown'
            })
        
        # Activity heatmap data (daily activity)
        activity_heatmap = {}
        for session in sessions:
            date_key = session.get('created_at', datetime.utcnow()).strftime('%Y-%m-%d') if hasattr(session.get('created_at'), 'strftime') else 'unknown'
            activity_heatmap[date_key] = activity_heatmap.get(date_key, 0) + 1
        
        # Skills radar chart data
        skills_data = {
            'writing': len([s for s in sessions if 'writing' in s.get('session_type', '')]),
            'vocabulary': len([e for e in events if 'vocabulary' in e.get('event_type', '')]),
            'stem': len([e for e in events if 'stem' in e.get('event_type', '')]),
            'isl_conversion': len([e for e in events if 'isl' in e.get('event_type', '')]),
            'sentence_practice': len([s for s in sessions if 'sentence' in s.get('session_type', '')])
        }
        
        # Progress pie chart
        session_types = {}
        for session in sessions:
            session_type = session.get('session_type', 'other')
            session_types[session_type] = session_types.get(session_type, 0) + 1
        
        return jsonify({
            'user_id': user_id,
            'charts': {
                'performance_line_chart': {
                    'title': 'Performance Over Time',
                    'data': performance_data,
                    'x_axis': 'session',
                    'y_axis': 'score'
                },
                'activity_heatmap': {
                    'title': 'Daily Learning Activity',
                    'data': activity_heatmap
                },
                'skills_radar_chart': {
                    'title': 'Skills Development',
                    'data': skills_data
                },
                'session_types_pie_chart': {
                    'title': 'Learning Activity Distribution',
                    'data': session_types
                }
            },
            'summary_stats': {
                'total_sessions': len(sessions),
                'total_events': len(events),
                'active_days': len(set(s.get('created_at', datetime.utcnow()).strftime('%Y-%m-%d') for s in sessions if hasattr(s.get('created_at'), 'strftime'))),
                'avg_daily_sessions': round(len(sessions) / max(1, len(set(s.get('created_at', datetime.utcnow()).strftime('%Y-%m-%d') for s in sessions if hasattr(s.get('created_at'), 'strftime')))), 2)
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Visual reports error: {str(e)}'}), 500

# Utility & System endpoints
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check database connection
        db_status = 'connected'
        try:
            client.admin.command('ping')
        except Exception:
            db_status = 'disconnected'
        
        # Check collections
        collections_status = {}
        try:
            collections = ['users', 'videos', 'practice_sessions', 'analytics_events']
            for collection_name in collections:
                count = db[collection_name].count_documents({})
                collections_status[collection_name] = f'{count} documents'
        except Exception as e:
            collections_status = {'error': str(e)}
        
        # System info
        system_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': 'running',
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
        
        # Overall health status
        overall_status = 'healthy' if db_status == 'connected' else 'unhealthy'
        
        return jsonify({
            'status': overall_status,
            'database': {
                'status': db_status,
                'uri': MONGO_URI.split('@')[1] if '@' in MONGO_URI else 'local',  # Hide credentials
                'collections': collections_status
            },
            'system': system_info,
            'services': {
                'authentication': 'active',
                'file_storage': 'active',
                'analytics': 'active',
                'ai_processing': 'active'
            }
        }), 200 if overall_status == 'healthy' else 503
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/feedback', methods=['POST'])
@token_required
def submit_feedback(current_user):
    data = request.get_json()
    
    feedback_type = data.get('type', 'general')  # bug, feature, general, improvement
    message = data.get('message')
    rating = data.get('rating')  # 1-5 stars
    category = data.get('category', 'general')  # ui, performance, content, functionality
    
    if not message:
        return jsonify({'error': 'Feedback message required hai'}), 400
    
    try:
        # Create feedback document
        feedback_doc = {
            'user_id': str(current_user['_id']),
            'username': current_user.get('username', 'anonymous'),
            'type': feedback_type,
            'category': category,
            'message': message,
            'rating': rating,
            'status': 'submitted',
            'created_at': datetime.utcnow(),
            'user_agent': request.headers.get('User-Agent', 'unknown'),
            'ip_address': request.remote_addr
        }
        
        # Store feedback
        result = feedback_collection.insert_one(feedback_doc)
        
        # Log feedback analytics
        analytics_event = AnalyticsEvent(
            user_id=str(current_user['_id']),
            event_type='feedback_submitted',
            event_data={
                'feedback_type': feedback_type,
                'category': category,
                'rating': rating,
                'message_length': len(message)
            }
        )
        analytics_events_collection.insert_one(analytics_event.to_dict())
        
        return jsonify({
            'message': 'Feedback successfully submitted!',
            'feedback_id': str(result.inserted_id),
            'status': 'received',
            'thank_you': 'Thank you for helping us improve our platform!'
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Feedback submission error: {str(e)}'}), 500


if __name__ == '__main__':
    print("Starting Flask app with Swagger documentation...")
    print("Swagger UI available at: http://localhost:5002/docs/")
    print("API Base URL: http://localhost:5002/")
    print(f"MongoDB URI: {MONGO_URI.split('@')[1] if '@' in MONGO_URI else 'local'}")
    print("\nAPI Endpoints Summary:")
    print("1. Authentication & User Management: 4 APIs")
    print("2. Alphabet & Numeracy Learning: 3 APIs")
    print("3. Vocabulary & Sentence Builder: 3 APIs")
    print("4. STEM Learning: 3 APIs")
    print("5. English <-> ISL Converter: 3 APIs")
    print("6. Learning Analytics Dashboard: 3 APIs")
    print("7. Utility & System: 2 APIs")
    print("\nTotal: 21 APIs Ready!")
    print("\n" + "="*50)
    app.run(debug=True, port=5002, host='0.0.0.0')