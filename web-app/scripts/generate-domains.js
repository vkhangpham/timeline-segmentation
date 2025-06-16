#!/usr/bin/env node
/**
 * Generate domains.json file automatically based on available data files
 * 
 * This script scans the public/data directory for comprehensive analysis files
 * and generates a domains.json file with all available domains.
 */

import fs from 'fs'
import path from 'path'
import process from 'process'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const DATA_DIR = path.join(__dirname, '..', 'public', 'data')
const OUTPUT_FILE = path.join(DATA_DIR, 'domains.json')

function generateDomainsJson() {
  try {
    console.log('🔍 Scanning for available domain data files...')
    
    // Read all files in the data directory
    const files = fs.readdirSync(DATA_DIR)
    console.log(`📁 Found ${files.length} files in data directory`)
    
    // Extract domain names from comprehensive analysis files
    const domains = new Set()
    const comprehensiveFiles = files.filter(file => 
      file.endsWith('_comprehensive_analysis.json')
    )
    
    console.log(`📊 Found ${comprehensiveFiles.length} comprehensive analysis files`)
    
    for (const file of comprehensiveFiles) {
      // Extract domain name (everything before _comprehensive_analysis.json)
      const domainName = file.replace('_comprehensive_analysis.json', '')
      domains.add(domainName)
      console.log(`✅ Discovered domain: ${domainName}`)
    }
    
    // Convert to sorted array
    const sortedDomains = Array.from(domains).sort()
    
    // Generate domains.json content
    const domainsData = {
      generated_at: new Date().toISOString(),
      total_count: sortedDomains.length,
      domains: sortedDomains,
      description: "Automatically generated from available comprehensive analysis data files",
      last_scan: {
        files_scanned: files.length,
        comprehensive_files_found: comprehensiveFiles.length,
        domains_discovered: sortedDomains.length
      }
    }
    
    // Write domains.json file
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(domainsData, null, 2))
    
    console.log(`\n🎉 Successfully generated domains.json with ${sortedDomains.length} domains:`)
    sortedDomains.forEach(domain => console.log(`   • ${domain}`))
    console.log(`💾 Saved to: ${OUTPUT_FILE}`)
    
    return true
    
  } catch (error) {
    console.error('❌ Error generating domains.json:', error)
    return false
  }
}

// Run the script
if (import.meta.url === `file://${process.argv[1]}`) {
  const success = generateDomainsJson()
  process.exit(success ? 0 : 1)
}

export { generateDomainsJson } 