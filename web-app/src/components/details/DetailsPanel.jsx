import { useState } from 'react'
import { X, Calendar } from 'lucide-react'
import PaperList from './PaperList'

function DetailsPanel({ segment, onClose }) {
  const [showAllPapers, setShowAllPapers] = useState(false)



  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between p-6 border-b border-gray-200">
        <div className="flex-1">
          <div className="flex items-center text-sm text-gray-500 mb-2">
            <Calendar className="w-4 h-4 mr-2" />
            <span>{segment.period[0]} - {segment.period[1]}</span>
          </div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">
            {segment.topic_label}
          </h2>
          <p className="text-gray-600 leading-relaxed">
            {segment.topic_description}
          </p>
        </div>
        <button
          onClick={onClose}
          className="ml-4 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>



      {/* Transition Indicators */}
      {segment.transition_indicators && segment.transition_indicators.length > 0 && (
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Transition Indicators</h3>
          <div className="flex flex-wrap gap-2">
            {segment.transition_indicators.map((indicator, index) => (
              <span
                key={index}
                className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
              >
                {indicator.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Representative Papers */}
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">
            Representative Papers ({segment.representative_papers.length})
          </h3>
          {segment.representative_papers.length > 5 && (
            <button
              onClick={() => setShowAllPapers(!showAllPapers)}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              {showAllPapers ? 'Show Less' : 'Show All'}
            </button>
          )}
        </div>
        
        <PaperList 
          papers={showAllPapers ? segment.representative_papers : segment.representative_papers.slice(0, 5)}
        />
        
        {!showAllPapers && segment.representative_papers.length > 5 && (
          <div className="mt-4 text-center">
            <button
              onClick={() => setShowAllPapers(true)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              View {segment.representative_papers.length - 5} more papers...
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default DetailsPanel 