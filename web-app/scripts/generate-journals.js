#!/usr/bin/env node

/**
 * Generate journals.json file automatically based on available journal HTML files
 * 
 * This script scans the public/journals directory for dev_journal_phaseN.html files
 * and generates a journals.json file with all available phases.
 */

import fs from 'fs'
import path from 'path'
import process from 'process'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const JOURNALS_DIR = path.join(__dirname, '..', 'public', 'journals')
const OUTPUT_FILE = path.join(JOURNALS_DIR, 'journals.json')

function generateJournalsJson() {
  try {
    console.log('🔍 Scanning for available journal files...')
    
    // Read all files in the journals directory
    const files = fs.readdirSync(JOURNALS_DIR)
    console.log(`📁 Found ${files.length} files in journals directory`)
    
    // Extract phase numbers from dev_journal_phaseN.html files
    const phases = new Set()
    const journalFiles = files.filter(file => 
      file.match(/^dev_journal_phase\d+\.html$/)
    )
    
    console.log(`📖 Found ${journalFiles.length} journal HTML files`)
    
    for (const file of journalFiles) {
      // Extract phase number from filename
      const match = file.match(/^dev_journal_phase(\d+)\.html$/)
      if (match) {
        const phaseNumber = parseInt(match[1])
        phases.add(phaseNumber)
        console.log(`✅ Discovered journal: Phase ${phaseNumber}`)
      }
    }
    
    // Convert to sorted array (latest phases first)
    const sortedPhases = Array.from(phases).sort((a, b) => b - a)
    
    // Generate phase data with metadata
    const phaseData = sortedPhases.map(phase => ({
      phase: phase.toString(),
      title: `Phase ${phase} Development Journal`,
      url: `/journals/dev_journal_phase${phase}.html`,
      filename: `dev_journal_phase${phase}.html`
    }))
    
    // Generate journals.json content
    const journalsData = {
      generated_at: new Date().toISOString(),
      total_count: sortedPhases.length,
      phases: phaseData,
      latest_phase: sortedPhases[0],
      available_phases: sortedPhases,
      description: "Automatically generated from available development journal HTML files",
      last_scan: {
        files_scanned: files.length,
        journal_files_found: journalFiles.length,
        phases_discovered: sortedPhases.length
      }
    }
    
    // Write journals.json file
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(journalsData, null, 2))
    
    console.log(`\n🎉 Successfully generated journals.json with ${sortedPhases.length} phases:`)
    sortedPhases.forEach(phase => console.log(`   • Phase ${phase}`))
    console.log(`📌 Latest phase: ${sortedPhases[0]}`)
    console.log(`💾 Saved to: ${OUTPUT_FILE}`)
    
    return true
    
  } catch (error) {
    console.error('❌ Error generating journals.json:', error)
    return false
  }
}

// Run the script
if (import.meta.url === `file://${process.argv[1]}`) {
  const success = generateJournalsJson()
  process.exit(success ? 0 : 1)
}

export { generateJournalsJson } 