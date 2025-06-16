import { useState, useEffect } from 'react'
import TimelineContainer from '../timeline/TimelineContainer'
import { fetchAvailableDomains, formatDomainName, isDomainDataAvailable } from '../../utils/domainUtils'

function TimelinePage() {
  const [domains, setDomains] = useState([])
  const [selectedDomain, setSelectedDomain] = useState('')
  const [timelineData, setTimelineData] = useState(null)
  const [groundTruthData, setGroundTruthData] = useState(null)
  const [loading, setLoading] = useState(false)

  // Fetch available domains on mount
  useEffect(() => {
    const loadDomains = async () => {
      try {
        // Fetch domains dynamically from generated domains.json
        const allDomains = await fetchAvailableDomains()
        
        // Verify each domain has data by testing for comprehensive analysis file
        const availableDomains = []
        for (const domain of allDomains) {
          const hasData = await isDomainDataAvailable(domain)
          if (hasData) {
            availableDomains.push(domain)
          } else {
            console.warn(`Domain ${domain} data not available`)
          }
        }
        
        console.log(`✅ Available domains with data: ${availableDomains.length}/${allDomains.length}`)
        console.log('Available domains:', availableDomains)
        
        setDomains(availableDomains)
        if (availableDomains.length > 0) {
          setSelectedDomain(availableDomains[0])
          console.log('Selected domain:', availableDomains[0])
        }
      } catch (error) {
        console.error('Error loading domains:', error)
        setDomains([])
      }
    }

    loadDomains()
  }, [])

  // Fetch timeline data when domain changes
  useEffect(() => {
    if (!selectedDomain) {
      console.log('No domain selected yet')
      return
    }

    console.log('Fetching timeline data for domain:', selectedDomain)

    const fetchTimelineData = async () => {
      setLoading(true)
      try {
        const url = `/data/${selectedDomain}_comprehensive_analysis.json`
        console.log('Fetching from URL:', url)
        
        const response = await fetch(url)
        console.log('Response status:', response.status, response.statusText)
        
        if (response.ok) {
          const data = await response.json()
          console.log('Timeline data loaded:', {
            hasMetadata: !!data.analysis_metadata,
            hasTimelineAnalysis: !!data.timeline_analysis,
            hasPeriodCharacterizations: !!data.timeline_analysis?.final_period_characterizations,
            segmentCount: data.timeline_analysis?.final_period_characterizations?.length
          })
          setTimelineData(data)
        } else {
          console.error('Failed to fetch timeline data for domain:', selectedDomain, response.status)
        }
      } catch (error) {
        console.error('Error fetching timeline data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchTimelineData()
  }, [selectedDomain])

  // Fetch ground truth data for current domain
  useEffect(() => {
    if (!selectedDomain) {
      setGroundTruthData(null)
      return
    }

    const fetchGroundTruth = async () => {
      try {
        const response = await fetch(`/validation/${selectedDomain}_groundtruth.json`)
        if (response.ok) {
          const data = await response.json()
          console.log('Ground truth data loaded for', selectedDomain, ':', data)
          setGroundTruthData(data)
        } else {
          console.warn('Ground truth data not available for', selectedDomain)
          setGroundTruthData(null)
        }
      } catch (error) {
        console.error('Error fetching ground truth data for', selectedDomain, ':', error)
        setGroundTruthData(null)
      }
    }
    
    fetchGroundTruth()
  }, [selectedDomain])

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900 mb-4">
          Research Timeline Analysis
        </h1>
        <p className="text-gray-600 mb-6">
          Explore the evolution of research paradigms and breakthrough moments across different domains.
        </p>
        
        {/* Domain Selector */}
        <div className="mb-6">
          <label htmlFor="domain-select" className="block text-sm font-medium text-gray-700 mb-2">
            Select Research Domain:
          </label>
          <select
            id="domain-select"
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
            className="block w-64 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            {domains.map(domain => (
              <option key={domain} value={domain}>
                {formatDomainName(domain)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Timeline Visualization */}
      {loading ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-500">Loading timeline data...</div>
          </div>
        </div>
      ) : timelineData ? (
        <div>
          <div className="text-sm text-gray-600 mb-6 bg-white rounded-lg border border-gray-200 p-4">
            <h2 className="text-xl font-semibold mb-2">
              {formatDomainName(selectedDomain)} Timeline
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="font-medium">Analysis period:</span> {timelineData.analysis_metadata?.time_range?.[0]} - {timelineData.analysis_metadata?.time_range?.[1]}
              </div>
              <div>
                <span className="font-medium">Total papers analyzed:</span> {timelineData.analysis_metadata?.total_papers_analyzed}
              </div>
              <div>
                <span className="font-medium">Research segments identified:</span> {timelineData.timeline_analysis?.final_period_characterizations?.length}
              </div>
            </div>
          </div>
          
          <TimelineContainer timelineData={timelineData} groundTruthData={groundTruthData} />
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-500">Select a domain to view timeline analysis</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default TimelinePage 