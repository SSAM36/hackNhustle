import { useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { authAPI } from '../services/api';
import BottomNav from './BottomNav';

export default function UserProfile() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({
    username: '',
    email: '',
    bio: ''
  });

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          navigate('/login');
          return;
        }

        const response = await authAPI.getProfile();
        setUser(response.data.user);
        setEditData({
          username: response.data.user.username || '',
          email: response.data.user.email || '',
          bio: response.data.user.bio || 'ISL learner passionate about sign language'
        });
      } catch (error) {
        console.error('Error fetching user data:', error);
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        navigate('/login');
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, []);

  const handleSave = async () => {
    try {
      // Update user data (mock for now)
      setUser({ ...user, ...editData });
      setIsEditing(false);
    } catch (error) {
      console.error('Error updating profile:', error);
    }
  };

  const handleCancel = () => {
    setEditData({
      username: user?.username || '',
      email: user?.email || '',
      bio: user?.bio || 'ISL learner passionate about sign language'
    });
    setIsEditing(false);
  };

  if (loading) {
    return (
      <div className="bg-background-light dark:bg-background-dark min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="bg-background-light dark:bg-background-dark text-slate-900 dark:text-white font-display antialiased selection:bg-primary selection:text-white pb-32 min-h-screen">
      {/* Top Navigation */}
      <header className="sticky top-0 z-40 bg-background-light/80 dark:bg-background-dark/80 backdrop-blur-md">
        <div className="flex items-center justify-between px-6 py-4">
          <button onClick={() => navigate('/')} className="flex items-center justify-center p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors">
            <span className="material-symbols-outlined text-slate-800 dark:text-white" style={{ fontSize: '24px' }}>arrow_back</span>
          </button>
          <h1 className="text-lg font-bold text-slate-800 dark:text-white tracking-tight">Profile</h1>
          <button 
            onClick={() => setIsEditing(!isEditing)}
            className="flex items-center justify-center p-2 rounded-full hover:bg-slate-200 dark:hover:bg-slate-800 transition-colors"
          >
            <span className="material-symbols-outlined text-slate-800 dark:text-white" style={{ fontSize: '24px' }}>
              {isEditing ? 'close' : 'edit'}
            </span>
          </button>
        </div>
      </header>

      <main className="flex flex-col gap-8 px-6 pt-2">
        {/* Profile Header */}
        <section className="flex flex-col items-center gap-4">
          <div className="relative group">
            <div className="h-32 w-32 rounded-full p-1 bg-gradient-to-tr from-primary to-emerald-300">
              <div className="h-full w-full rounded-full bg-cover bg-center border-4 border-white dark:border-background-dark" style={{ backgroundImage: "url('https://lh3.googleusercontent.com/aida-public/AB6AXuC6fIhh1GH2J18RPPDdcJhMckQ4KBJszie17DB4uoh6FKFbNoypPk4Q57kNtL1drAEfRmdlZjfKBFytMo1pNbAEbtr_LG9MqfadlFqQrfgZ4aM_znSE-mT0LmMtF0p1xosPGtGkQg0LyaPKzKlSk2B-9MtWRyPWhUgas7afJ-PnUTAlxn-BIeV83PPP_FOA6PaLxoPro7s66-cgehFRY6rm0dF1_Pe-2dRQ4J_l6YoHZYtMNUSqy70CSeAoMGUDJQFufPU3PBiUzKlJ')" }} />
            </div>
            <div className="absolute bottom-0 right-0 bg-white dark:bg-slate-800 rounded-full p-2 shadow-lg border-2 border-background-light dark:border-background-dark">
              <span className="material-symbols-outlined text-primary text-xl font-bold block">verified</span>
            </div>
          </div>
          <div className="text-center space-y-2">
            {isEditing ? (
              <div className="space-y-3">
                <input
                  type="text"
                  value={editData.username}
                  onChange={(e) => setEditData({...editData, username: e.target.value})}
                  className="text-3xl font-extrabold text-primary dark:text-emerald-400 tracking-tight bg-transparent border-b-2 border-primary text-center focus:outline-none"
                  placeholder="Username"
                />
                <input
                  type="email"
                  value={editData.email}
                  onChange={(e) => setEditData({...editData, email: e.target.value})}
                  className="text-sm text-slate-500 dark:text-slate-400 bg-transparent border-b border-slate-300 text-center focus:outline-none w-full"
                  placeholder="Email"
                />
                <textarea
                  value={editData.bio}
                  onChange={(e) => setEditData({...editData, bio: e.target.value})}
                  className="text-sm text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-700 rounded-lg p-3 w-full resize-none focus:outline-none focus:ring-2 focus:ring-primary"
                  placeholder="Bio"
                  rows="2"
                />
                <div className="flex gap-2 justify-center">
                  <button
                    onClick={handleSave}
                    className="px-4 py-2 bg-primary text-white rounded-full text-sm font-semibold hover:bg-primary/90 transition-colors"
                  >
                    Save
                  </button>
                  <button
                    onClick={handleCancel}
                    className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-800 dark:text-white rounded-full text-sm font-semibold hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <>
                <h2 className="text-3xl font-extrabold text-primary dark:text-emerald-400 tracking-tight">{user?.username || 'User'}</h2>
                <p className="text-sm text-slate-600 dark:text-slate-300">{user?.email}</p>
                <div className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-primary/10 dark:bg-primary/20 text-primary dark:text-emerald-300 text-sm font-semibold">
                  <span className="material-symbols-outlined text-[18px]">workspace_premium</span>
                  Level 12 Signer
                </div>
                <p className="text-slate-500 dark:text-slate-400 text-sm">{editData.bio}</p>
                <p className="text-slate-500 dark:text-slate-400 text-xs">Member since 2023</p>
              </>
            )}
          </div>
        </section>

        {/* Stats Bento Box */}
        <section className="grid grid-cols-2 gap-4">
          {/* Learning Streak */}
          <div className="col-span-2 bg-white dark:bg-slate-800 p-6 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 flex flex-row items-center justify-between relative overflow-hidden group">
            <div className="flex flex-col gap-2 z-10">
              <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 font-medium">
                <span className="material-symbols-outlined text-coral">local_fire_department</span>
                <span>Learning Streak</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-extrabold text-slate-900 dark:text-white">12</span>
                <span className="text-lg font-bold text-slate-500 dark:text-slate-400">Days</span>
              </div>
              <p className="text-sm text-green-600 dark:text-green-400 font-medium flex items-center gap-1">
                <span className="material-symbols-outlined text-sm">trending_up</span>
                Personal Best!
              </p>
            </div>
            <span className="material-symbols-outlined absolute -right-4 -bottom-4 text-[120px] text-coral/5 dark:text-coral/10 rotate-12 pointer-events-none">local_fire_department</span>
          </div>

          {/* Signs Learned */}
          <div className="col-span-1 bg-white dark:bg-slate-800 p-5 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col gap-3">
            <div className="h-10 w-10 rounded-xl bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400">
              <span className="material-symbols-outlined">sign_language</span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900 dark:text-white block">124</span>
              <span className="text-sm font-medium text-slate-500 dark:text-slate-400">Signs Learned</span>
            </div>
          </div>

          {/* Accuracy */}
          <div className="col-span-1 bg-white dark:bg-slate-800 p-5 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700 flex flex-col gap-3">
            <div className="h-10 w-10 rounded-xl bg-purple-50 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400">
              <span className="material-symbols-outlined">target</span>
            </div>
            <div>
              <span className="text-2xl font-bold text-slate-900 dark:text-white block">92%</span>
              <span className="text-sm font-medium text-slate-500 dark:text-slate-400">Accuracy</span>
            </div>
          </div>
        </section>

        {/* Learning Progress Chart */}
        <section className="bg-white dark:bg-slate-800 p-6 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-700">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Learning Progress</h3>
            <div className="flex gap-1 bg-slate-100 dark:bg-slate-700 p-1 rounded-lg">
              <button className="px-3 py-1 rounded-md bg-white dark:bg-slate-600 shadow-sm text-xs font-bold text-slate-900 dark:text-white">Week</button>
              <button className="px-3 py-1 rounded-md text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white text-xs font-medium transition-colors">Month</button>
            </div>
          </div>
          <div className="flex flex-col gap-1 mb-6">
            <span className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">3.5 hrs</span>
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-500 dark:text-slate-400">Total this week</span>
              <span className="text-xs font-bold text-green-600 bg-green-100 dark:bg-green-900/30 dark:text-green-400 px-2 py-0.5 rounded-full">+12%</span>
            </div>
          </div>
          <div className="w-full h-40 relative">
            <svg className="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 350 100">
              <defs>
                <linearGradient id="gradient" x1="0%" x2="0%" y1="0%" y2="100%">
                  <stop offset="0%" stopColor="#0f756d" stopOpacity="0.2" />
                  <stop offset="100%" stopColor="#0f756d" stopOpacity="0" />
                </linearGradient>
              </defs>
              <path d="M0,80 Q35,70 70,50 T140,40 T210,60 T280,30 T350,20 V100 H0 Z" fill="url(#gradient)" />
              <path d="M0,80 Q35,70 70,50 T140,40 T210,60 T280,30 T350,20" fill="none" stroke="#0f756d" strokeLinecap="round" strokeLinejoin="round" strokeWidth="4" />
              <circle className="fill-white dark:fill-slate-800" cx="350" cy="20" r="6" stroke="#0f756d" strokeWidth="3" />
            </svg>
          </div>
          <div className="flex justify-between mt-4 text-xs font-medium text-slate-400 dark:text-slate-500">
            <span>Mon</span>
            <span>Tue</span>
            <span>Wed</span>
            <span>Thu</span>
            <span>Fri</span>
            <span>Sat</span>
            <span className="text-primary font-bold">Sun</span>
          </div>
        </section>

        {/* Achievements */}
        <section>
          <div className="flex items-center justify-between mb-4 px-1">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Achievements</h3>
            <button className="text-primary text-sm font-semibold hover:opacity-80">View all</button>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-4 bg-white dark:bg-slate-800 p-4 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700">
              <div className="h-12 w-12 rounded-xl bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined text-amber-600 dark:text-amber-500 text-2xl">speed</span>
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-slate-900 dark:text-white">Fast Learner</h4>
                <p className="text-sm text-slate-500 dark:text-slate-400">Completed 10 lessons in 1 hour</p>
              </div>
              <span className="material-symbols-outlined text-slate-300 dark:text-slate-600">chevron_right</span>
            </div>
            <div className="flex items-center gap-4 bg-white dark:bg-slate-800 p-4 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700">
              <div className="h-12 w-12 rounded-xl bg-pink-100 dark:bg-pink-900/30 flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined text-pink-600 dark:text-pink-500 text-2xl">volunteer_activism</span>
              </div>
              <div className="flex-1">
                <h4 className="font-bold text-slate-900 dark:text-white">Community Hero</h4>
                <p className="text-sm text-slate-500 dark:text-slate-400">Helped 5 peers with signs</p>
              </div>
              <span className="material-symbols-outlined text-slate-300 dark:text-slate-600">chevron_right</span>
            </div>
          </div>
        </section>
      </main>

      {/* Bottom Navigation */}
      <BottomNav />
    </div>
  );
}
