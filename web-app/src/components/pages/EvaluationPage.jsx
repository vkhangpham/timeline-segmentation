import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Info, AlertTriangle, AlertCircle } from 'lucide-react'
import { fetchAvailableDomains, formatDomainName } from '../../utils/domainUtils'

function EvaluationPage() {
  const [evaluationResults, setEvaluationResults] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchEvaluationResults = async () => {
      try {
        // Dynamically discover available domains and their evaluation results
        const domains = await fetchAvailableDomains()
        console.log(`🔍 Looking for evaluation results for ${domains.length} domains`)
        
        const results = []
        for (const domain of domains) {
          const filename = `${domain}_evaluation_results.json`
          try {
            const response = await fetch(`/validation/${filename}`)
            if (response.ok) {
              const data = await response.json()
              results.push({
                domain: domain,
                filename: filename,
                data: data
              })
              console.log(`✅ Loaded evaluation results for ${domain}`)
            } else {
              console.warn(`⚠️ No evaluation results found for ${domain}`)
            }
          } catch (error) {
            console.error(`❌ Error fetching ${filename}:`, error)
          }
        }
        
        console.log(`📊 Successfully loaded ${results.length}/${domains.length} evaluation results`)
        setEvaluationResults(results)
      } catch (error) {
        console.error('Error fetching evaluation results:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchEvaluationResults()
  }, [])

  const formatMetricValue = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(3)
    }
    return String(value)
  }

  const getAssessmentIcon = (assessment) => {
    if (!assessment) return <Info className="w-5 h-5 text-gray-400" />
    
    if (assessment.includes('GOOD') || assessment.includes('✅')) {
      return <CheckCircle className="w-5 h-5 text-green-500" />
    } else if (assessment.includes('POOR') || assessment.includes('❌')) {
      return <XCircle className="w-5 h-5 text-red-500" />
    } else if (assessment.includes('FAIR') || assessment.includes('⚠️')) {
      return <AlertTriangle className="w-5 h-5 text-yellow-500" />
    }
    return <AlertCircle className="w-5 h-5 text-blue-500" />
  }

  const getAssessmentColor = (assessment) => {
    if (!assessment) return 'bg-gray-50 border-gray-200 text-gray-800'
    
    if (assessment.includes('GOOD') || assessment.includes('✅')) {
      return 'bg-green-50 border-green-200 text-green-800'
    } else if (assessment.includes('POOR') || assessment.includes('❌')) {
      return 'bg-red-50 border-red-200 text-red-800'
    } else if (assessment.includes('FAIR') || assessment.includes('⚠️')) {
      return 'bg-yellow-50 border-yellow-200 text-yellow-800'
    }
    return 'bg-blue-50 border-blue-200 text-blue-800'
  }

  const MetricCard = ({ title, value, description }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-900">{title}</h3>
        <Info className="w-4 h-4 text-gray-400" />
      </div>
      <div className="text-2xl font-semibold text-gray-900 mb-1">
        {formatMetricValue(value)}
      </div>
      {description && (
        <p className="text-sm text-gray-500">{description}</p>
      )}
    </div>
  )

  const getMetrics = (data) => {
    // Handle different data structures
    if (data.metrics) {
      return data.metrics
    }
    if (data.recall_evaluation?.metrics) {
      return data.recall_evaluation.metrics
    }
    return null
  }

  const getLLMSummary = (data) => {
    return data.enhanced_llm_evaluation?.summary || null
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-gray-900 mb-4">
          Evaluation Results
        </h1>
        <p className="text-gray-600">
          Performance metrics and validation results for timeline analysis across different domains.
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading evaluation results...</div>
        </div>
      ) : evaluationResults.length > 0 ? (
        <div className="space-y-8">
          {evaluationResults.map(({ domain, data, filename }) => {
            const metrics = getMetrics(data)
            const llmSummary = getLLMSummary(data)
            const assessment = data.assessment
            
            return (
              <div key={domain} className="bg-white rounded-lg shadow-sm border border-gray-200">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-xl font-semibold text-gray-900">
                    {formatDomainName(domain)}
                  </h2>
                  <p className="text-sm text-gray-500 mt-1">Source: {filename}</p>
                </div>
                
                <div className="p-6">
                  {/* Overall Assessment - Prominent Display */}
                  {assessment && (
                    <div className="mb-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4">Overall Assessment</h3>
                      <div className={`border rounded-lg p-4 flex items-center gap-3 ${getAssessmentColor(assessment)}`}>
                        {getAssessmentIcon(assessment)}
                        <p className="text-sm font-medium">{assessment}</p>
                      </div>
                    </div>
                  )}

                  {/* Sanity Check Status */}
                  {data.sanity_check_passed !== undefined && (
                    <div className="mb-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4">Sanity Check</h3>
                      <div className={`border rounded-lg p-4 flex items-center gap-3 ${
                        data.sanity_check_passed 
                          ? 'bg-green-50 border-green-200 text-green-800'
                          : 'bg-red-50 border-red-200 text-red-800'
                      }`}>
                        {data.sanity_check_passed 
                          ? <CheckCircle className="w-5 h-5 text-green-500" />
                          : <XCircle className="w-5 h-5 text-red-500" />
                        }
                        <p className="text-sm font-medium">
                          {data.sanity_check_passed ? 'Passed' : 'Failed'}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Core Metrics */}
                  {metrics && (
                    <div className="mb-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4">Core Metrics</h3>
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <MetricCard
                          title="Precision"
                          value={metrics.precision}
                          description="Accuracy of detected segments"
                        />
                        <MetricCard
                          title="Recall"
                          value={metrics.recall}
                          description="Coverage of ground truth periods"
                        />
                        <MetricCard
                          title="F1 Score"
                          value={metrics.f1_score}
                          description="Balanced precision-recall measure"
                        />
                        <MetricCard
                          title="Matches"
                          value={`${metrics.true_positives || 0}/${(metrics.true_positives || 0) + (metrics.false_negatives || 0)}`}
                          description="True positives / Total ground truth"
                        />
                      </div>
                    </div>
                  )}

                  {/* LLM Evaluation Summary */}
                  {llmSummary && (
                    <div className="mb-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4">Enhanced LLM Evaluation</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <MetricCard
                          title="LLM Precision"
                          value={llmSummary.precision}
                          description={`${llmSummary.valid_segments || 0} of ${llmSummary.total_segments || 0} segments valid`}
                        />
                        <MetricCard
                          title="Models Used"
                          value={`${llmSummary.models_successful || 0}/${llmSummary.models_attempted || 0}`}
                          description="Successful model evaluations"
                        />
                        <MetricCard
                          title="Three Pillar Labels"
                          value={llmSummary.three_pillar_labels_used ? "Yes" : "No"}
                          description="Enhanced labeling system used"
                        />
                      </div>
                      
                      {llmSummary.criteria_metrics && (
                        <div>
                          <h4 className="text-md font-medium text-gray-900 mb-3">Quality Criteria</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {Object.entries(llmSummary.criteria_metrics).map(([key, value]) => (
                              <div key={key} className="bg-gray-50 rounded-lg p-3 text-center">
                                <div className="text-lg font-semibold text-gray-900">{value}</div>
                                <div className="text-xs text-gray-600 mt-1">
                                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Matching Results Summary */}
                  {(data.matching_results || data.recall_evaluation?.matching_results) && (
                    <div className="mb-6">
                      <h3 className="text-lg font-medium text-gray-900 mb-4">Matching Analysis</h3>
                      {(() => {
                        const matchingResults = data.matching_results || data.recall_evaluation.matching_results
                        return (
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <MetricCard
                              title="Matched Segments"
                              value={matchingResults.match_count || 0}
                              description={`Out of ${matchingResults.total_algo_segments || 0} algorithm segments`}
                            />
                            <MetricCard
                              title="Unmatched Algorithm"
                              value={matchingResults.unmatched_algorithm?.length || 0}
                              description="Algorithm segments without ground truth match"
                            />
                            <MetricCard
                              title="Unmatched Ground Truth"
                              value={matchingResults.unmatched_ground_truth?.length || 0}
                              description="Ground truth periods not detected"
                            />
                          </div>
                        )
                      })()}
                    </div>
                  )}

                  {/* Raw JSON Expandable */}
                  <details className="mt-6">
                    <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                      View Raw Evaluation Data
                    </summary>
                    <pre className="mt-3 text-xs bg-gray-50 rounded-lg p-4 overflow-x-auto">
                      {JSON.stringify(data, null, 2)}
                    </pre>
                  </details>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
          <XCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Evaluation Results Found</h3>
          <p className="text-gray-500">
            No evaluation result files were found in the validation directory.
          </p>
        </div>
      )}
    </div>
  )
}

export default EvaluationPage 