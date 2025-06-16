import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import { BookOpen, FileText, Search } from 'lucide-react'

function ResearchPage() {
  const [documents, setDocuments] = useState([])
  const [selectedDoc, setSelectedDoc] = useState('')
  const [docContent, setDocContent] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const fetchDocuments = async () => {
      const knownDocs = [
        { 
          filename: 'Time Series Segmentation.md',
          title: 'Time Series Segmentation of Scientific Literature',
          description: 'Comprehensive overview of methodologies for segmenting time series data in scientific publications'
        }
      ]
      
      const availableDocs = []
      for (const doc of knownDocs) {
        try {
          const response = await fetch(`/docs/${encodeURIComponent(doc.filename)}`)
          if (response.ok) {
            availableDocs.push(doc)
          }
        } catch (error) {
          console.error(`Error checking ${doc.filename}:`, error)
        }
      }
      
      setDocuments(availableDocs)
      if (availableDocs.length > 0) {
        setSelectedDoc(availableDocs[0].filename)
      }
    }

    fetchDocuments()
  }, [])

  useEffect(() => {
    if (!selectedDoc) return

    const fetchDocContent = async () => {
      setLoading(true)
      try {
        const response = await fetch(`/docs/${encodeURIComponent(selectedDoc)}`)
        if (response.ok) {
          const content = await response.text()
          setDocContent(content)
        } else {
          setDocContent('Failed to load document content.')
        }
      } catch (error) {
        console.error('Error fetching document content:', error)
        setDocContent('Error loading document content.')
      } finally {
        setLoading(false)
      }
    }

    fetchDocContent()
  }, [selectedDoc])

  const selectedDocInfo = documents.find(doc => doc.filename === selectedDoc)

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-80 bg-white shadow-sm border-r border-gray-200 min-h-screen">
          <div className="p-6">
            <div className="flex items-center mb-6">
              <BookOpen className="w-5 h-5 mr-2 text-blue-600" />
              <h2 className="text-lg font-semibold text-gray-900">Research Documentation</h2>
            </div>
            
            {documents.length > 0 ? (
              <nav className="space-y-3">
                {documents.map(doc => (
                  <button
                    key={doc.filename}
                    onClick={() => setSelectedDoc(doc.filename)}
                    className={`w-full text-left p-4 rounded-lg border transition-colors duration-200 ${
                      selectedDoc === doc.filename
                        ? 'bg-blue-50 border-blue-200 text-blue-900'
                        : 'bg-white border-gray-200 text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-start">
                      <FileText className="w-4 h-4 mr-3 mt-1 flex-shrink-0" />
                      <div>
                        <div className="font-medium text-sm leading-5">
                          {doc.title}
                        </div>
                        <div className="text-xs text-gray-500 mt-2 leading-4">
                          {doc.description}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </nav>
            ) : (
              <div className="text-center text-gray-500 py-8">
                <Search className="w-8 h-8 mx-auto mb-3" />
                <p className="text-sm">No research documents found</p>
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1">
          <div className="max-w-5xl mx-auto px-8 py-8">
            {selectedDocInfo && (
              <div className="mb-8">
                <div className="flex items-center text-sm text-gray-500 mb-3">
                  <BookOpen className="w-4 h-4 mr-2" />
                  <span>Research Documentation</span>
                </div>
                <h1 className="text-3xl font-semibold text-gray-900 mb-2">
                  {selectedDocInfo.title}
                </h1>
                <p className="text-gray-600 text-lg">
                  {selectedDocInfo.description}
                </p>
              </div>
            )}

            {loading ? (
              <div className="flex items-center justify-center h-64">
                <div className="flex items-center text-gray-500">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-500 mr-3"></div>
                  Loading document content...
                </div>
              </div>
            ) : docContent ? (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                <div className="prose prose-gray prose-lg max-w-none p-10 markdown-content">
                  <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                    {docContent}
                  </ReactMarkdown>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
                <BookOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-medium text-gray-900 mb-3">No Document Selected</h3>
                <p className="text-gray-500 text-lg">
                  Select a research document from the sidebar to view its content.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResearchPage 