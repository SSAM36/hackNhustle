from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is working!"

@app.route('/create-user', methods=['POST'])
def create_user():
    data = request.get_json()
    return jsonify({'message': 'User created!', 'data': data})

if __name__ == '__main__':
    app.run(debug=True, port=5002)