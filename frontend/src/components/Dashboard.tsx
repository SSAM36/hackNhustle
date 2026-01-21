import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { authAPI } from '../services/api'
import BottomNav from './BottomNav'

export default function Dashboard() {
  const navigate = useNavigate()
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token')
        const storedUser = localStorage.getItem('user')
        
        if (!token) {
          navigate('/login')
          return
        }

        // First try to use stored user data
        if (storedUser) {
          try {
            const userData = JSON.parse(storedUser)
            setUser(userData)
            setLoading(false)
            return
          } catch (e) {
            console.log('Invalid stored user data, fetching from API')
          }
        }

        // If no stored user or invalid data, fetch from API
        const response = await authAPI.getProfile()
        setUser(response.data.user)
        localStorage.setItem('user', JSON.stringify(response.data.user))
      } catch (error) {
        console.error('Error fetching user data:', error)
        // Don't immediately redirect on API error, try stored data first
        const storedUser = localStorage.getItem('user')
        if (storedUser) {
          try {
            const userData = JSON.parse(storedUser)
            setUser(userData)
            setLoading(false)
            return
          } catch (e) {
            // If stored data is also invalid, then redirect
            localStorage.removeItem('token')
            localStorage.removeItem('user')
            navigate('/login')
          }
        } else {
          localStorage.removeItem('token')
          localStorage.removeItem('user')
          navigate('/login')
        }
      } finally {
        setLoading(false)
      }
    }

    fetchUserData()
  }, [])

  const getGreeting = () => {
    const hour = new Date().getHours()
    if (hour < 12) return 'Good Morning'
    if (hour < 17) return 'Good Afternoon'
    return 'Good Evening'
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  if (loading) {
    return (
      <div className="w-full max-w-md bg-background-light dark:bg-background-dark relative flex flex-col h-screen overflow-hidden items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        <p className="mt-4 text-slate-600 dark:text-slate-400">Loading...</p>
      </div>
    )
  }
  return (
    <div className="w-full max-w-md bg-background-light dark:bg-background-dark relative flex flex-col h-screen overflow-hidden">
      {/* Main Content Scroll Area */}
      <main className="flex-1 overflow-y-auto overflow-x-hidden p-5 pb-28 space-y-5 hide-scrollbar relative z-10">
        {/* Floating Pill Header */}
        <header className="flex items-center justify-between bg-white dark:bg-surface-dark rounded-full p-2 pl-4 pr-2 shadow-sm border border-slate-100 dark:border-slate-800 sticky top-0 z-20">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-slate-200 bg-cover bg-center border-2 border-white dark:border-slate-700 shadow-sm" style={{ backgroundImage: "url('https://lh3.googleusercontent.com/aida-public/AB6AXuAIwt2hFBU6I0cc0sLC2SgI5SSOjR6FuDMAA9T8r4NBJeLnEOsJsf9zks-1bAoRj2SOaavh3Tf6K3lwKTF4NWrtPgVGyQ02yEDspUBXqVLMXIimd7O6E5wRx9rJa7CD5rLF4TZCyZt-UuTB5thBKa-MqUZ2m-YA30j0FlamSysBZV2NTfxAvCc-gP9ayFfx2ASEzeXmGMljedp03lgBsVbhX1qPCaiFriH8Bu9mGOs0ULRajy8A8FGmWoWPBtIYySOYPFAM7zsb8JAm')" }} />
            <div>
              <h1 className="text-sm font-bold text-slate-800 dark:text-white leading-tight">
                {getGreeting()}, {user?.username || 'User'}
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">Ready to learn?</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="bg-[#FFF1F2] dark:bg-[#FB7185]/20 px-3 py-1.5 rounded-full flex items-center gap-1.5 border border-[#FB7185]/20">
              <span className="material-symbols-outlined text-[#FB7185] text-[18px]">local_fire_department</span>
              <span className="text-[#FB7185] font-bold text-sm">12</span>
            </div>
            <button 
              onClick={() => navigate('/sign-to-word')}
              className="p-2 rounded-full bg-primary text-white hover:bg-primary-dark transition-colors shadow-lg"
              title="Scan Signs"
            >
              <span className="material-symbols-outlined text-[18px]">videocam</span>
            </button>
            <button 
              onClick={handleLogout}
              className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
              title="Logout"
            >
              <span className="material-symbols-outlined text-slate-600 dark:text-slate-400 text-[18px]">logout</span>
            </button>
          </div>
        </header>

        {/* Hero 'Active Lesson' Card */}
        <section className="w-full bg-white dark:bg-surface-dark rounded-3xl p-6 shadow-soft relative overflow-hidden group">
          <div className="absolute -right-10 -top-10 w-40 h-40 bg-primary/5 rounded-full blur-3xl" />
          <div className="flex justify-between items-start mb-6 relative z-10">
            <div className="flex flex-col gap-1">
              <span className="text-primary font-semibold text-sm tracking-wide uppercase">Continue Learning</span>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white leading-tight">Unit 1:<br />Basic Greetings</h2>
              <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">Next: Saying Goodbye</p>
            </div>
            {/* Circular Progress */}
            <div className="relative size-16 flex items-center justify-center">
              <svg className="size-full transform -rotate-90" viewBox="0 0 36 36">
                <path className="text-slate-100 dark:text-slate-700" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="3" />
                <path className="text-primary drop-shadow-md" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeDasharray="75, 100" strokeLinecap="round" strokeWidth="3" />
              </svg>
              <span className="absolute text-xs font-bold text-slate-800 dark:text-white">75%</span>
            </div>
          </div>
          <button 
            onClick={() => navigate('/lessons')}
            className="w-full bg-slate-900 dark:bg-primary text-white dark:text-slate-900 h-12 rounded-full font-semibold text-sm flex items-center justify-center gap-2 hover:bg-slate-800 dark:hover:bg-primary-dark transition-colors relative z-10 shadow-lg shadow-slate-200 dark:shadow-none cursor-pointer"
          >
            <span>Continue Lesson</span>
            <span className="material-symbols-outlined text-sm">arrow_forward</span>
          </button>
        </section>

        {/* Communication Core (Bento Grid) */}
        <section className="grid grid-cols-2 gap-4 h-64">
          {/* Text-to-Sign Portal */}
          <div onClick={() => navigate('/translate')} className="bg-primary rounded-3xl p-5 flex flex-col justify-between relative overflow-hidden shadow-glow shadow-primary/20 group cursor-pointer transition-transform active:scale-[0.98]">
            <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent" />
            <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center text-white relative z-10">
              <span className="material-symbols-outlined">keyboard</span>
            </div>
            <div className="relative z-10">
              <h3 className="text-white text-lg font-bold leading-tight mb-1">Text to<br />Sign</h3>
              <p className="text-white/80 text-xs font-medium">Type to Translate</p>
            </div>
            <div className="absolute -right-4 -bottom-4 opacity-10 rotate-12">
              <span className="material-symbols-outlined text-[100px] text-white">waving_hand</span>
            </div>
          </div>

          {/* Sign-to-Text Portal */}
          <div onClick={() => navigate('/sign-to-word')}  className="bg-white dark:bg-surface-dark border-2 border-primary/20 rounded-3xl p-5 flex flex-col justify-between relative overflow-hidden shadow-soft cursor-pointer transition-all active:scale-[0.98] hover:border-primary/50">
            <div className="w-12 h-12 bg-slate-50 dark:bg-slate-700 rounded-full flex items-center justify-center text-primary relative z-10">
              <span className="material-symbols-outlined">videocam</span>
            </div>
            <div className="relative z-10">
              <h3 className="text-slate-900 dark:text-white text-lg font-bold leading-tight mb-1">Sign to<br />Text</h3>
              <p className="text-slate-500 dark:text-slate-400 text-xs font-medium">Scan Hand Signs</p>
            </div>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-32 h-32 border border-primary/10 rounded-full animate-pulse" />
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-20 h-20 border border-primary/20 rounded-full" />
          </div>
        </section>

        {/* Tools Heading */}
        <div className="flex items-center justify-between px-1">
          <h3 className="text-lg font-bold text-slate-900 dark:text-white">Daily Practice</h3>
          <button className="text-xs font-semibold text-primary">View All</button>
        </div>

        {/* Horizontal Scroll Tools */}
        <section className="flex gap-4 overflow-x-auto hide-scrollbar pb-4 -mx-5 px-5">
          {[
            { icon: 'school', title: 'STEM Lessons', subtitle: 'Learn Math & Science', color: 'green', route: '/lessons' },
            { icon: 'edit', title: 'Practice', subtitle: 'Tracing Pad', color: 'blue', route: '/tracing' }
          ].map((tool, i) => (
            <div 
              key={i} 
              onClick={() => tool.route && navigate(tool.route)}
              className="min-w-[140px] aspect-square bg-white dark:bg-surface-dark rounded-3xl p-4 flex flex-col items-center justify-center gap-3 shadow-soft border border-slate-50 dark:border-slate-800 text-center relative overflow-hidden cursor-pointer hover:scale-105 transition-transform"
            >
              {tool.badge && <div className="absolute top-2 right-2 w-2 h-2 bg-accent rounded-full" />}
              <div className="w-12 h-12 rounded-2xl flex items-center justify-center" style={{
                backgroundColor: tool.color === 'orange' ? '#fed7aa' : 
                                tool.color === 'purple' ? '#e9d5ff' : 
                                tool.color === 'blue' ? '#dbeafe' : 
                                tool.color === 'green' ? '#dcfce7' : '#d1fae5',
                color: tool.color === 'orange' ? '#ea580c' : 
                       tool.color === 'purple' ? '#9333ea' : 
                       tool.color === 'blue' ? '#2563eb' : 
                       tool.color === 'green' ? '#16a34a' : '#059669'
              }}>
                <span className="material-symbols-outlined">{tool.icon}</span>
              </div>
              <div>
                <p className="font-bold text-slate-800 dark:text-white text-sm">{tool.title}</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">{tool.subtitle}</p>
              </div>
            </div>
          ))}
        </section>
      </main>

      {/* Floating Glassmorphic Dock */}
      <BottomNav />
    </div>
  );
}
