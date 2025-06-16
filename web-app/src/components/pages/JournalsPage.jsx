import { useState, useEffect } from 'react'
import { FileText, Calendar, ExternalLink, Download } from 'lucide-react'
import { fetchAvailableJournals, isJournalAvailable } from '../../utils/journalUtils'

function JournalsPage() {
  const [availablePhases, setAvailablePhases] = useState([])
  const [selectedPhase, setSelectedPhase] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const loadJournals = async () => {
      setLoading(true)
      try {
        // Fetch journals dynamically from generated journals.json
        const allJournals = await fetchAvailableJournals()
        
        // Verify each journal has data by testing for HTML file
        const availableJournals = []
        for (const journal of allJournals) {
          const hasData = await isJournalAvailable(journal.phase)
          if (hasData) {
            availableJournals.push(journal)
          } else {
            console.warn(`Journal phase ${journal.phase} data not available`)
          }
        }
        
        console.log(`✅ Available journals with data: ${availableJournals.length}/${allJournals.length}`)
        console.log('Available journals:', availableJournals.map(j => `Phase ${j.phase}`))
        
        setAvailablePhases(availableJournals)
        if (availableJournals.length > 0) {
          setSelectedPhase(availableJournals[0].phase) // Default to latest phase (first in sorted list)
          console.log('Selected journal:', availableJournals[0].phase)
        }
      } catch (error) {
        console.error('Error loading journals:', error)
        setAvailablePhases([])
      } finally {
        setLoading(false)
      }
    }

    loadJournals()
  }, [])

  const selectedPhaseData = availablePhases.find(p => p.phase === selectedPhase)

  const openInNewTab = () => {
    if (selectedPhaseData) {
      window.open(selectedPhaseData.url, '_blank')
    }
  }

  const printJournal = () => {
    if (selectedPhaseData) {
      const iframe = document.getElementById('journal-iframe')
      if (iframe) {
        iframe.contentWindow.print()
      }
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
          <div className="p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Development Journals</h2>
            
            {loading ? (
              <div className="text-sm text-gray-500">Loading journals...</div>
            ) : availablePhases.length > 0 ? (
              <nav className="space-y-2">
                {availablePhases.map(phase => (
                  <button
                    key={phase.phase}
                    onClick={() => setSelectedPhase(phase.phase)}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors duration-200 ${
                      selectedPhase === phase.phase
                        ? 'bg-blue-50 text-blue-700 border border-blue-200'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center">
                      <FileText className="w-4 h-4 mr-2" />
                      <div>
                        <div className="font-medium">Phase {phase.phase}</div>
                        <div className="text-xs text-gray-500 mt-1">Development Journal</div>
                      </div>
                    </div>
                  </button>
                ))}
              </nav>
            ) : (
              <div className="text-sm text-gray-500">No journals found</div>
            )}

            {/* Action buttons */}
            {selectedPhaseData && (
              <div className="mt-6 pt-6 border-t border-gray-200 space-y-2">
                <button
                  onClick={openInNewTab}
                  className="w-full flex items-center px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-md transition-colors"
                >
                  <ExternalLink className="w-4 h-4 mr-2" />
                  Open in New Tab
                </button>
                <button
                  onClick={printJournal}
                  className="w-full flex items-center px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-md transition-colors"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Print/Save as PDF
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1">
          {selectedPhaseData ? (
            <div className="h-screen flex flex-col">
              {/* Header */}
              <div className="bg-white border-b border-gray-200 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center text-sm text-gray-500 mb-1">
                      <Calendar className="w-4 h-4 mr-2" />
                      <span>Development Journal</span>
                    </div>
                    <h1 className="text-2xl font-semibold text-gray-900">
                      Phase {selectedPhase} Development Journal
                    </h1>
                  </div>
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={openInNewTab}
                      className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Open in New Tab
                    </button>
                    <button
                      onClick={printJournal}
                      className="inline-flex items-center px-3 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 transition-colors"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Print/Save PDF
                    </button>
                  </div>
                </div>
              </div>

              {/* Journal Content */}
              <div className="flex-1 bg-white">
                <iframe
                  id="journal-iframe"
                  src={selectedPhaseData.url}
                  className="w-full h-full border-0"
                  title={`Phase ${selectedPhase} Development Journal`}
                  style={{ minHeight: 'calc(100vh - 80px)' }}
                />
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-screen">
              <div className="text-center">
                <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-xl font-medium text-gray-900 mb-2">
                  {loading ? 'Loading Journals...' : 'No Journal Selected'}
                </h3>
                <p className="text-gray-500 max-w-sm">
                  {loading 
                    ? 'Checking available development journals...'
                    : 'Select a development phase from the sidebar to view the professional journal documentation.'
                  }
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default JournalsPage 