import { useState } from 'react'
import { ExternalLink, Star, Calendar, Quote, ChevronDown, ChevronUp, Hash } from 'lucide-react'

function PaperList({ papers }) {
  const [expandedPaper, setExpandedPaper] = useState(null)

  const formatCitationCount = (count) => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`
    }
    return count.toLocaleString()
  }

  const truncateTitle = (title, maxLength = 80) => {
    if (title.length <= maxLength) return title
    return title.substring(0, maxLength) + '...'
  }

  const getKeywordTags = (keywords, maxTags = 3) => {
    if (!keywords || keywords.length === 0) return []
    return keywords.slice(0, maxTags)
  }

  const cleanAbstract = (abstract, title, keywords) => {
    if (!abstract) return ''
    
    let cleaned = abstract
    
    // Remove title from beginning if it appears there
    if (title && cleaned.startsWith(title)) {
      cleaned = cleaned.substring(title.length).trim()
    }
    
    // Remove duplicate title patterns
    const titleVariations = [
      title,
      title + '\n',
      title + '\n\n'
    ].filter(Boolean)
    
    titleVariations.forEach(variation => {
      if (cleaned.startsWith(variation)) {
        cleaned = cleaned.substring(variation.length).trim()
      }
    })
    
    // Remove keywords from the end if they appear there
    if (keywords && keywords.length > 0) {
      const keywordText = keywords.join(', ')
      const keywordTextAlt = keywords.join(',')
      
      // Check for various keyword patterns at the end
      const keywordPatterns = [
        keywordText,
        keywordTextAlt,
        '\n' + keywordText,
        '\n\n' + keywordText,
        '\n' + keywordTextAlt,
        '\n\n' + keywordTextAlt
      ]
      
      keywordPatterns.forEach(pattern => {
        if (cleaned.endsWith(pattern)) {
          cleaned = cleaned.substring(0, cleaned.length - pattern.length).trim()
        }
      })
    }
    
    return cleaned.trim()
  }

  return (
    <>
      <div className="space-y-4">
        {papers.map((paper, index) => {
          const isExpanded = expandedPaper === paper.openalex_id
          return (
            <div
              key={paper.openalex_id || index}
              className="border border-gray-200 rounded-lg bg-white hover:shadow-sm transition-shadow"
            >
              <div className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    {/* Paper Title */}
                    <div className="flex items-start mb-2">
                      <h4 className="text-lg font-medium text-gray-900 leading-6 pr-4">
                        {isExpanded ? paper.title : truncateTitle(paper.title)}
                      </h4>
                      {paper.breakthrough_status && (
                        <Star className="w-4 h-4 text-amber-500 flex-shrink-0 mt-1" />
                      )}
                    </div>

                    {/* Metadata */}
                    <div className="flex items-center text-sm text-gray-500 mb-3 space-x-4">
                      <div className="flex items-center">
                        <Calendar className="w-3 h-3 mr-1" />
                        <span>{paper.year}</span>
                      </div>
                      <div className="flex items-center">
                        <Quote className="w-3 h-3 mr-1" />
                        <span>{formatCitationCount(paper.citation_count)} citations</span>
                      </div>
                      {paper.breakthrough_status && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800">
                          Breakthrough
                        </span>
                      )}
                    </div>

                    {/* Keywords */}
                    {paper.keywords && paper.keywords.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-3">
                        {(isExpanded ? paper.keywords : getKeywordTags(paper.keywords)).map((keyword, keyIndex) => (
                          <span
                            key={keyIndex}
                            className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800"
                          >
                            {keyword}
                          </span>
                        ))}
                        {!isExpanded && paper.keywords.length > 3 && (
                          <span className="text-xs text-gray-500 px-2 py-1">
                            +{paper.keywords.length - 3} more
                          </span>
                        )}
                      </div>
                    )}

                    {/* Abstract Preview/Full */}
                    {paper.abstract && (
                      <div className="text-sm text-gray-600 leading-5 mb-3">
                        {isExpanded ? (
                          <div className="prose prose-sm max-w-none">
                            <p className="whitespace-pre-line">{cleanAbstract(paper.abstract, paper.title, paper.keywords)}</p>
                          </div>
                        ) : (
                          <p className="line-clamp-2">
                            {(() => {
                              const cleaned = cleanAbstract(paper.abstract, paper.title, paper.keywords)
                              return cleaned.length > 200 
                                ? cleaned.substring(0, 200) + '...'
                                : cleaned
                            })()}
                          </p>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex flex-col space-y-2 ml-4">
                    <button
                      onClick={() => setExpandedPaper(isExpanded ? null : paper.openalex_id)}
                      className="inline-flex items-center px-3 py-1.5 border border-gray-300 text-xs font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 transition-colors"
                    >
                      {isExpanded ? (
                        <>
                          <ChevronUp className="w-3 h-3 mr-1" />
                          Less Details
                        </>
                      ) : (
                        <>
                          <ChevronDown className="w-3 h-3 mr-1" />
                          More Details
                        </>
                      )}
                    </button>
                    {paper.openalex_id && (
                      <a
                        href={paper.openalex_id}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-blue-600 hover:text-blue-800 transition-colors"
                      >
                        <ExternalLink className="w-3 h-3 mr-1" />
                        OpenAlex
                      </a>
                    )}
                  </div>
                </div>
              </div>


            </div>
          )
        })}
      </div>

    </>
  )
}

export default PaperList 