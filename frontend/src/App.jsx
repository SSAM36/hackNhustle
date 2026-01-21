import Dashboard from './components/Dashboard'
import Translator from './components/Translator'
import SignToWord from './components/SignToWord'
import UserProfile from './components/UserProfile'
import Login from './components/Login'
import Signup from './components/Signup'
import Practice from './components/Practice'
import TracingPad from './components/TracingPad'
import DailyQuest from './components/DailyQuest'
import Dictionary from './components/Dictionary'
import Flashcards from './components/Flashcards'
import Lessons from './components/Lessons'
import { Routes, Route } from 'react-router-dom'
import './index.css'

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/" element={<Dashboard />} />
      <Route path="/translate" element={<Translator />} />
      <Route path="/sign-to-word" element={<SignToWord />} />
      <Route path="/lessons" element={<Lessons />} />
      <Route path="/flashcards" element={<Flashcards />} />
      <Route path="/dictionary" element={<Dictionary />} />
      <Route path="/daily-quest" element={<DailyQuest />} />
      <Route path="/practice" element={<Practice />} />
      <Route path="/tracing" element={<TracingPad />} />
      <Route path="/profile" element={<UserProfile />} />
      <Route path="/stats" element={<Dashboard />} />
    </Routes>
  )
}

export default App
