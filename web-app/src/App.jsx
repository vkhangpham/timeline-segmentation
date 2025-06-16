import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { FileText, BarChart3, BookOpen, FlaskConical } from 'lucide-react'
import TimelinePage from './components/pages/TimelinePage'
import EvaluationPage from './components/pages/EvaluationPage'
import JournalsPage from './components/pages/JournalsPage'
import ResearchPage from './components/pages/ResearchPage'

function Navigation() {
  const location = useLocation()
  
  const navItems = [
    { path: '/', label: 'Timeline', icon: BarChart3 },
    { path: '/evaluation', label: 'Evaluation', icon: FlaskConical },
    { path: '/journals', label: 'Dev Journals', icon: FileText },
    { path: '/research', label: 'Research', icon: BookOpen },
  ]

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <h1 className="text-xl font-semibold text-gray-900">
              Timeline Analysis System
            </h1>
          </div>
          <div className="flex space-x-8">
            {navItems.map(({ path, label, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200 ${
                  location.pathname === path
                    ? 'border-blue-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4 mr-2" />
                {label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  )
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main>
          <Routes>
            <Route path="/" element={<TimelinePage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/journals" element={<JournalsPage />} />
            <Route path="/research" element={<ResearchPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
