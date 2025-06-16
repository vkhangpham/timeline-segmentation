/**
 * Journal Utilities
 * 
 * Utilities for fetching and managing available development journals dynamically
 */

/**
 * Fetch available journals from the generated journals.json file
 * 
 * @returns {Promise<Array>} Array of journal phase objects
 */
export async function fetchAvailableJournals() {
  try {
    const response = await fetch('/journals/journals.json')
    
    if (!response.ok) {
      console.warn('Failed to fetch journals.json, falling back to hardcoded phases')
      // Fallback to known phases if the file is not available
      return generateFallbackPhases([9, 8, 7, 6, 5, 4, 3, 2, 1])
    }
    
    const data = await response.json()
    console.log(`✅ Dynamically loaded ${data.total_count} journal phases from journals.json`)
    console.log(`📅 Generated at: ${data.generated_at}`)
    console.log(`📌 Latest phase: ${data.latest_phase}`)
    
    return data.phases || []
    
  } catch (error) {
    console.error('Error fetching journals:', error)
    
    // Fallback to known phases on error
    const fallbackPhases = generateFallbackPhases([9, 8, 7, 6, 5, 4, 3, 2, 1])
    
    console.warn(`Using fallback phases: ${fallbackPhases.map(p => p.phase).join(', ')}`)
    return fallbackPhases
  }
}

/**
 * Generate fallback phase data
 * 
 * @param {number[]} phaseNumbers - Array of phase numbers
 * @returns {Array} Array of phase objects
 */
function generateFallbackPhases(phaseNumbers) {
  return phaseNumbers.map(phase => ({
    phase: phase.toString(),
    title: `Phase ${phase} Development Journal`,
    url: `/journals/dev_journal_phase${phase}.html`,
    filename: `dev_journal_phase${phase}.html`
  }))
}

/**
 * Check if journal data is available by testing for the HTML file
 * 
 * @param {string} phase - Phase number as string
 * @returns {Promise<boolean>} True if journal is available
 */
export async function isJournalAvailable(phase) {
  try {
    const response = await fetch(`/journals/dev_journal_phase${phase}.html`, { method: 'HEAD' })
    return response.ok
  } catch {
    return false
  }
}

/**
 * Get the latest available phase
 * 
 * @returns {Promise<string|null>} Latest phase number as string, or null if none available
 */
export async function getLatestPhase() {
  try {
    const journals = await fetchAvailableJournals()
    return journals.length > 0 ? journals[0].phase : null
  } catch {
    return null
  }
} 