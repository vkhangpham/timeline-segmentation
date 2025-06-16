import { X, ExternalLink, Star, Calendar, Quote, Hash } from 'lucide-react'

function PaperModal({ paper, onClose }) {
  const formatCitationCount = (count) => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`
    }
    return count.toLocaleString()
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-start justify-between p-6 border-b border-gray-200 sticky top-0 bg-white">
          <div className="flex-1 min-w-0">
            <div className="flex items-start mb-2">
              <h2 className="text-xl font-semibold text-gray-900 leading-7 pr-4">
                {paper.title}
              </h2>
              {paper.breakthrough_status && (
                <Star className="w-5 h-5 text-amber-500 flex-shrink-0 mt-1" />
              )}
            </div>
            
            <div className="flex items-center text-sm text-gray-500 space-x-4">
              <div className="flex items-center">
                <Calendar className="w-4 h-4 mr-1" />
                <span>{paper.year}</span>
              </div>
              <div className="flex items-center">
                <Quote className="w-4 h-4 mr-1" />
                <span>{formatCitationCount(paper.citation_count)} citations</span>
              </div>
              {paper.breakthrough_status && (
                <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-amber-100 text-amber-800">
                  <Star className="w-3 h-3 mr-1" />
                  Breakthrough Paper
                </span>
              )}
            </div>
          </div>
          
          <button
            onClick={onClose}
            className="ml-4 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Abstract */}
          {paper.abstract && (
            <div className="mb-6">
              <h3 className="text-lg font-medium text-gray-900 mb-3">Abstract</h3>
              <div className="prose prose-gray max-w-none">
                <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                  {paper.abstract}
                </p>
              </div>
            </div>
          )}

          {/* Keywords */}
          {paper.keywords && paper.keywords.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                <Hash className="w-4 h-4 mr-2" />
                Keywords & Topics ({paper.keywords.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {paper.keywords.map((keyword, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Paper Details */}
          <div className="mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Paper Details</h3>
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium text-gray-700">Publication Year</div>
                  <div className="text-gray-900">{paper.year}</div>
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-700">Citation Count</div>
                  <div className="text-gray-900">{paper.citation_count.toLocaleString()}</div>
                </div>
              </div>
              
              {paper.openalex_id && (
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-1">OpenAlex ID</div>
                  <div className="text-gray-900 font-mono text-sm break-all">
                    {paper.openalex_id}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* External Links */}
          <div className="flex space-x-3">
            {paper.openalex_id && (
              <a
                href={paper.openalex_id}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                View on OpenAlex
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default PaperModal 