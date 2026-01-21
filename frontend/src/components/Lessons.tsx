import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../services/api';

const lessonsData = {
  maths: {
    id: 'maths',
    title: 'Mathematics',
    icon: '🔢',
    color: 'bg-blue-500',
    chapters: [
      {
        id: 'shapes',
        title: 'Shapes and Space',
        videoId: 'BGBS-CFn3YA',
        duration: '15 min'
      },
      {
        id: 'numbers',
        title: 'Numbers',
        videoId: '27kD7WunkHI',
        duration: '12 min'
      },
      {
        id: 'numbers_b',
        title: 'Numbers Part B',
        videoId: 'ejkKO0CauSg',
        duration: '10 min'
      },
      {
        id: 'addition',
        title: 'Addition',
        videoId: '8jywPyKp364',
        duration: '18 min'
      },
      {
        id: 'addition_b',
        title: 'Addition Part B',
        videoId: 'WR6LXNOYwUU',
        duration: '14 min'
      },
      {
        id: 'addition_c',
        title: 'Addition Part C',
        videoId: 'x916cBZzqLQ',
        duration: '16 min'
      },
      {
        id: 'subtraction',
        title: 'Subtraction',
        videoId: 'IAhrwiLCiFI',
        duration: '20 min'
      },
      {
        id: 'subtraction_b',
        title: 'Subtraction Part B',
        videoId: 'K49siiPUjU4',
        duration: '15 min'
      }
    ]
  },
  science: {
    id: 'science',
    title: 'Science',
    icon: '🔬',
    color: 'bg-green-500',
    chapters: [
      {
        id: 'weather',
        title: 'Weather',
        videoId: '1oOYUsdBOO4',
        duration: '22 min'
      },
      {
        id: 'climate',
        title: 'Climate',
        videoId: 'w-cFolXQohk',
        duration: '18 min'
      },
      {
        id: 'wet_climate',
        title: 'Wet Climate',
        videoId: '2nt7Ob0fHuc',
        duration: '16 min'
      },
      {
        id: 'dry_climate',
        title: 'Dry Climate',
        videoId: 'GGbMlckEPLM',
        duration: '20 min'
      },
      {
        id: 'digestive_system',
        title: 'Digestive System',
        videoId: 'Mnuy4dNPUEU',
        duration: '25 min'
      },
      {
        id: 'circulatory_system',
        title: 'Circulatory System',
        videoId: '49jbiZOwRI4',
        duration: '23 min'
      }
    ]
  }
};

export default function Lessons() {
  const navigate = useNavigate();
  const [selectedSubject, setSelectedSubject] = useState('maths');
  const [userProgress, setUserProgress] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUserProgress();
  }, []);

  const fetchUserProgress = async () => {
    try {
      const response = await authAPI.getUserProgress();
      // Convert backend format to frontend format
      const backendProgress = response.data.progress || {};
      const frontendProgress = {};
      
      Object.keys(backendProgress).forEach(subject => {
        frontendProgress[subject] = backendProgress[subject].completed_chapters || [];
      });
      
      setUserProgress(frontendProgress);
    } catch (error) {
      console.error('Error fetching progress:', error);
      setUserProgress({});
    } finally {
      setLoading(false);
    }
  };

  const markChapterComplete = async (subjectId, chapterId) => {
    try {
      await authAPI.markChapterComplete(subjectId, chapterId);
      
      // Update local state
      const newProgress = { ...userProgress };
      if (!newProgress[subjectId]) {
        newProgress[subjectId] = [];
      }
      if (!newProgress[subjectId].includes(chapterId)) {
        newProgress[subjectId].push(chapterId);
      }
      setUserProgress(newProgress);
      
      // Show success message
      alert('Chapter completed! 🎉');
    } catch (error) {
      console.error('Error marking chapter complete:', error);
      alert('Error saving progress. Please try again.');
    }
  };

  const getSubjectProgress = (subjectId) => {
    const subject = lessonsData[subjectId];
    const completedChapters = userProgress[subjectId] || [];
    return Math.round((completedChapters.length / subject.chapters.length) * 100);
  };

  const isChapterCompleted = (subjectId, chapterId) => {
    return userProgress[subjectId]?.includes(chapterId) || false;
  };

  if (loading) {
    return (
      <div className="bg-background-light dark:bg-background-dark h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  const currentSubject = lessonsData[selectedSubject];

  return (
    <div className="bg-background-light dark:bg-background-dark h-screen flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between p-4 pt-6">
        <button onClick={() => navigate('/')} className="flex size-10 items-center justify-center rounded-full hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors">
          <span className="material-symbols-outlined text-slate-800 dark:text-white">arrow_back</span>
        </button>
        <h1 className="text-lg font-bold text-slate-800 dark:text-white">STEM Lessons</h1>
        <div className="size-10" />
      </header>

      {/* Subject Selector */}
      <div className="px-4 mb-6">
        <div className="flex gap-4">
          {Object.values(lessonsData).map((subject) => (
            <button
              key={subject.id}
              onClick={() => setSelectedSubject(subject.id)}
              className={`flex-1 p-4 rounded-2xl transition-all ${
                selectedSubject === subject.id
                  ? `${subject.color} text-white shadow-lg`
                  : 'bg-white dark:bg-slate-800 text-slate-800 dark:text-white hover:bg-slate-50 dark:hover:bg-slate-700'
              }`}
            >
              <div className="text-2xl mb-2">{subject.icon}</div>
              <div className="font-bold text-sm">{subject.title}</div>
              <div className="text-xs opacity-80 mt-1">
                {getSubjectProgress(subject.id)}% Complete
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Progress Overview */}
      <div className="px-4 mb-6">
        <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm border border-slate-100 dark:border-slate-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-bold text-slate-800 dark:text-white">{currentSubject.title} Progress</h3>
            <span className="text-sm text-slate-500 dark:text-slate-400">
              {(userProgress[selectedSubject] || []).length}/{currentSubject.chapters.length} chapters
            </span>
          </div>
          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${currentSubject.color}`}
              style={{ width: `${getSubjectProgress(selectedSubject)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Chapters List */}
      <div className="flex-1 px-4 pb-4 overflow-y-auto">
        <div className="space-y-4">
          {currentSubject.chapters.map((chapter, index) => (
            <ChapterCard
              key={chapter.id}
              chapter={chapter}
              index={index}
              subjectId={selectedSubject}
              isCompleted={isChapterCompleted(selectedSubject, chapter.id)}
              onMarkComplete={markChapterComplete}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function ChapterCard({ chapter, index, subjectId, isCompleted, onMarkComplete }) {
  const [showVideo, setShowVideo] = useState(false);

  return (
    <div className="bg-white dark:bg-slate-800 rounded-2xl p-4 shadow-sm border border-slate-100 dark:border-slate-700">
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-bold text-slate-500 dark:text-slate-400">
              Chapter {index + 1}
            </span>
            {isCompleted && (
              <span className="material-symbols-outlined text-green-500 text-lg">check_circle</span>
            )}
          </div>
          <h3 className="font-bold text-slate-800 dark:text-white mb-1">{chapter.title}</h3>
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
              <span className="material-symbols-outlined text-sm">schedule</span>
              {chapter.duration}
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
              <span className="material-symbols-outlined text-sm">play_circle</span>
              ISL Video
            </span>
          </div>
        </div>
      </div>

      {showVideo && (
        <div className="mb-4">
          <div className="aspect-video rounded-xl overflow-hidden bg-slate-100 dark:bg-slate-700">
            <iframe
              width="100%"
              height="100%"
              src={`https://www.youtube.com/embed/${chapter.videoId}`}
              title={chapter.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
              allowFullScreen
            />
          </div>
        </div>
      )}

      <div className="flex gap-2">
        <button
          onClick={() => setShowVideo(!showVideo)}
          className="flex-1 bg-primary text-white py-3 rounded-xl font-semibold hover:bg-primary/90 transition-colors"
        >
          {showVideo ? 'Hide Video' : 'Watch Lesson'}
        </button>
        
        {showVideo && !isCompleted && (
          <button
            onClick={() => onMarkComplete(subjectId, chapter.id)}
            className="px-4 py-3 bg-green-500 text-white rounded-xl font-semibold hover:bg-green-600 transition-colors"
          >
            ✓ Complete
          </button>
        )}
      </div>
    </div>
  );
}