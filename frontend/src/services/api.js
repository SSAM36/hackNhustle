import axios from 'axios';

const API_BASE_URL = 'http://localhost:5002';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add JWT token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Authentication API
export const authAPI = {
  register: (userData) => api.post('/auth/register', userData),
  login: (credentials) => api.post('/auth/login', credentials),
  getRole: () => api.get('/auth/role'),
  getProfile: () => api.get('/user/profile'),
  getUserProgress: () => api.get('/user/progress'),
  markChapterComplete: (subjectId, chapterId) => api.post('/user/progress/chapter', {
    subject_id: subjectId,
    chapter_id: chapterId
  }),
  recognizeSign: (imageData, topK = 1) => api.post('/convert/sign-to-text', {
    image_data: imageData,
    confidence_threshold: 0.9,
    top_k: topK
  }),
};

// Conversion API
export const convertAPI = {
  textToSign: (data) => api.post('/convert/text-to-sign', data),
  signToText: (data) => api.post('/convert/sign-to-text', data),
  speechToSign: (data) => api.post('/convert/speech-to-sign', data),
};

// System API
export const systemAPI = {
  healthCheck: () => api.get('/health'),
  submitFeedback: (data) => api.post('/feedback', data),
};

export default api;