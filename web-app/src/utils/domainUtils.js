/**
 * Domain Utilities
 * 
 * Utilities for fetching and managing available domains dynamically
 */

/**
 * Fetch available domains from the generated domains.json file
 * 
 * @returns {Promise<string[]>} Array of domain names
 */
export async function fetchAvailableDomains() {
  try {
    const response = await fetch('/data/domains.json')
    
    if (!response.ok) {
      console.warn('Failed to fetch domains.json, falling back to hardcoded domains')
      // Fallback to known domains if the file is not available
      return [
        'applied_mathematics',
        'art', 
        'computer_science',
        'computer_vision',
        'deep_learning',
        'machine_learning',
        'machine_translation',
        'natural_language_processing'
      ]
    }
    
    const data = await response.json()
    console.log(`✅ Dynamically loaded ${data.total_count} domains from domains.json`)
    console.log(`📅 Generated at: ${data.generated_at}`)
    
    return data.domains || []
    
  } catch (error) {
    console.error('Error fetching domains:', error)
    
    // Fallback to known domains on error
    const fallbackDomains = [
      'applied_mathematics',
      'art', 
      'computer_science',
      'computer_vision',
      'deep_learning',
      'machine_learning',
      'machine_translation',
      'natural_language_processing'
    ]
    
    console.warn(`Using fallback domains: ${fallbackDomains.join(', ')}`)
    return fallbackDomains
  }
}

/**
 * Format domain name for display (capitalize words, replace underscores)
 * 
 * @param {string} domain - Domain name to format
 * @returns {string} Formatted domain name
 */
export function formatDomainName(domain) {
  return domain
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase())
}

/**
 * Check if domain data is available by testing for a key file
 * 
 * @param {string} domain - Domain name to check
 * @returns {Promise<boolean>} True if domain data is available
 */
export async function isDomainDataAvailable(domain) {
  try {
    // Test for comprehensive analysis file as it's usually the main output
    const response = await fetch(`/data/${domain}_comprehensive_analysis.json`)
    return response.ok
  } catch {
    return false
  }
} 