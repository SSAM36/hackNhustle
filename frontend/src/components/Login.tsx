import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../services/api';

export default function Login() {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({ username: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const response = await authAPI.login({
        username: formData.username,
        password: formData.password
      });
      
      // Store token and user data
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
      
      // Navigate to dashboard
      navigate('/');
    } catch (error: any) {
      setError(error.response?.data?.error || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="font-display bg-background-light dark:bg-background-dark text-[#111717] antialiased selection:bg-primary/30 h-screen overflow-hidden">
      <div className="relative h-full flex flex-col w-full overflow-x-hidden">
        {/* Header Section */}
        <header className="relative h-[42vh] w-full bg-primary flex flex-col items-center pt-12 overflow-hidden">
          {/* Decorative Background Elements */}
          <div className="absolute top-0 left-0 w-full h-full overflow-hidden opacity-20 pointer-events-none">
            <div className="absolute -top-10 -right-10 w-64 h-64 bg-white rounded-full mix-blend-overlay filter blur-3xl" />
            <div className="absolute top-20 -left-20 w-80 h-80 bg-[#4fd1c5] rounded-full mix-blend-overlay filter blur-3xl" />
          </div>
          
          {/* Illustration / Brand */}
          <div className="relative z-10 flex flex-col items-center gap-6 animate-fade-in px-6 text-center">
            <div className="p-4 bg-white/10 backdrop-blur-md rounded-full shadow-inner ring-1 ring-white/20">
              <span className="material-symbols-outlined text-white text-5xl" style={{ fontVariationSettings: "'FILL' 1" }}>
                diversity_2
              </span>
            </div>
            <div className="flex flex-col items-center">
              <h1 className="text-white text-3xl font-bold tracking-tight">SignLingo</h1>
              <p className="text-white/60 text-sm font-medium tracking-widest uppercase mt-1">Learn together</p>
            </div>
            
            {/* Abstract Hands Illustration */}
            <div className="mt-4 w-full max-w-[280px] h-32 relative opacity-80">
              <img 
                alt="Abstract hands connecting against a teal background" 
                className="w-full h-full object-cover rounded-2xl" 
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCEnfeX-ntCYEbHzS8jnE2Ta5ebJq79ehbs7d1m0uSBcmWwZJDMBmwMQn49M1aYIhWMoFeouc6lnAd6cn6KPl1lA0YXWrqAudX6HxP_BF-jc-ndvLcMaPKZV_1FeXXY8G4fXGi18tlnsMoA4pESvWqMLQJzAlpe5yDfq0t5HNI-BK6ViS3CTpub6PnlRXgyb__TKN-5Q2-LKOd80stkYyuKFC_2afY3Xqh2wXa1lOcTNspA4_TDTAHQMYBWJsdkSk0VqdJsb3heBgim" 
                style={{ WebkitMaskImage: 'linear-gradient(to bottom, black 50%, transparent 100%)', maskImage: 'linear-gradient(to bottom, black 50%, transparent 100%)' }}
              />
            </div>
          </div>
        </header>

        {/* Action Panel */}
        <main className="flex-1 bg-white dark:bg-[#1a2c2b] rounded-t-[2.5rem] -mt-10 relative z-20 flex flex-col shadow-[0_-10px_40px_-15px_rgba(0,0,0,0.1)] transition-colors duration-300">
          <div className="w-full max-w-md mx-auto p-6 pt-8 pb-8 flex flex-col h-full">
            {/* Greeting */}
            <div className="mb-6 space-y-1">
              <h2 className="text-2xl font-bold text-[#111717] dark:text-white">Welcome Back!</h2>
              <p className="text-gray-500 dark:text-gray-400 text-sm">Let's continue your signing journey.</p>
            </div>

            {/* Login Form */}
            <form className="space-y-5 flex-1" onSubmit={handleSubmit}>
              {/* Email Input */}
              <div className="space-y-2">
                <label className="text-xs font-bold uppercase tracking-wider text-gray-400 ml-1">Username</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-5 flex items-center pointer-events-none">
                    <span className="material-symbols-outlined text-gray-400 group-focus-within:text-primary">person</span>
                  </div>
                  <input 
                    name="username"
                    value={formData.username}
                    onChange={handleInputChange}
                    className="block w-full pl-12 pr-5 py-4 bg-background-light dark:bg-black/20 border-transparent rounded-xl text-base text-[#111717] dark:text-white placeholder-gray-400 focus:border-primary/50 focus:bg-white dark:focus:bg-black/30 focus:ring-4 focus:ring-primary/10 transition-all duration-200" 
                    placeholder="Enter your username" 
                    type="text"
                    required
                  />
                </div>
              </div>

              {/* Password Input */}
              <div className="space-y-2">
                <div className="flex justify-between items-center ml-1">
                  <label className="text-xs font-bold uppercase tracking-wider text-gray-400">Password</label>
                </div>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-5 flex items-center pointer-events-none">
                    <span className="material-symbols-outlined text-gray-400 group-focus-within:text-primary">lock</span>
                  </div>
                  <input 
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    className="block w-full pl-12 pr-12 py-4 bg-background-light dark:bg-black/20 border-transparent rounded-xl text-base text-[#111717] dark:text-white placeholder-gray-400 focus:border-primary/50 focus:bg-white dark:focus:bg-black/30 focus:ring-4 focus:ring-primary/10 transition-all duration-200" 
                    placeholder="••••••••" 
                    type={showPassword ? 'text' : 'password'}
                    required
                  />
                  <button 
                    className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors" 
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    <span className="material-symbols-outlined text-xl">{showPassword ? 'visibility' : 'visibility_off'}</span>
                  </button>
                </div>
                <div className="text-right mt-1">
                  <a className="text-xs font-semibold text-primary hover:text-primary/80 transition-colors" href="#">Forgot Password?</a>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg text-sm">
                  {error}
                </div>
              )}

              {/* Primary Action */}
              <div className="pt-2">
                <button 
                  className="w-full py-4 bg-primary hover:bg-teal-700 text-white font-bold text-lg rounded-full shadow-[0_8px_20px_-6px_rgba(15,117,109,0.4)] transform hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed" 
                  type="submit"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Logging in...</span>
                    </>
                  ) : (
                    <>
                      <span>Log In</span>
                      <span className="material-symbols-outlined text-xl">arrow_forward</span>
                    </>
                  )}
                </button>
              </div>

              {/* Divider */}
              <div className="relative flex items-center py-2">
                <div className="flex-grow border-t border-gray-100 dark:border-gray-700" />
                <span className="flex-shrink-0 mx-4 text-gray-400 text-xs font-medium">Or continue with</span>
                <div className="flex-grow border-t border-gray-100 dark:border-gray-700" />
              </div>

              {/* Social Logins */}
              <div className="grid grid-cols-2 gap-4">
                <button className="flex items-center justify-center gap-3 py-3 px-4 rounded-full border border-gray-200 dark:border-gray-700 bg-white dark:bg-transparent hover:bg-gray-50 dark:hover:bg-white/5 transition-colors group" type="button">
                  <img alt="Google logo" className="w-5 h-5" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCkddvYXCR3FkG3r6hO0FFFXy9EhtkqoKK0U547Z8MwNFeeN4EE3sO3tHir5OxUcAFiKgXWnm0QiToBthk8Ib-a07XM8UsDWfqsSj8v-0FaXlyKx1EyLDXyQcadmT6Pon5hbPuj5AODQ7W_sdUE228PDS4hQN2AxS_vYhE3FJNd4FhZAPSvH5IBMm1vhHNXsb0PgL8GN5muZ_xmpDjA9t3PR1Q5xsEiFIyKlXRjDL0iAvdtg88koKk6RUeObIl0xYhlKa3-_YaHOjTO" />
                  <span className="text-sm font-semibold text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white">Google</span>
                </button>
                <button className="flex items-center justify-center gap-3 py-3 px-4 rounded-full border border-gray-200 dark:border-gray-700 bg-white dark:bg-transparent hover:bg-gray-50 dark:hover:bg-white/5 transition-colors group" type="button">
                  <span className="material-symbols-outlined text-2xl text-black dark:text-white">ios</span>
                  <span className="text-sm font-semibold text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white">Apple</span>
                </button>
              </div>
            </form>

            {/* Footer */}
            <div className="mt-8 text-center pb-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                New to SignLingo? 
                <a onClick={() => navigate('/signup')} className="font-bold text-coral hover:text-red-500 transition-colors ml-1 inline-flex items-center gap-0.5 group cursor-pointer">
                  Create an Account
                  <span className="material-symbols-outlined text-sm transition-transform group-hover:translate-x-0.5">chevron_right</span>
                </a>
              </p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
