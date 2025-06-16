#!/bin/bash
#
# Timeline Analysis Web Application Launcher
#
# This script automatically parses development journals and launches the interactive
# timeline visualization web application with all Phase 6 enhancements.
#
# Usage:
#     ./run_web_app.sh
#     ./run_web_app.sh --parse-only
#     ./run_web_app.sh --dev-only
#     ./run_web_app.sh --help
#

set -e  # Exit on any error

# Colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_APP_DIR="${PROJECT_ROOT}/web-app"
PARSE_SCRIPT="${PROJECT_ROOT}/scripts/parse_dev_journals.py"
PARSED_OUTPUT_DIR="${WEB_APP_DIR}/public/parsed_journals"

# Default options
PARSE_ONLY=false
DEV_ONLY=false
SHOW_HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parse-only)
            PARSE_ONLY=true
            shift
            ;;
        --dev-only)
            DEV_ONLY=true
            shift
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display help information
if [ "$SHOW_HELP" = true ]; then
    echo -e "${CYAN}🚀 TIMELINE ANALYSIS WEB APPLICATION LAUNCHER${NC}"
    echo -e "${CYAN}Phase 6 Enhanced Visualization with Development Journal Integration${NC}"
    echo "======================================================================"
    echo ""
    echo "This script automates the development workflow by:"
    echo "• Parsing all development journal files into structured JSON"
    echo "• Launching the React web application with live data integration"
    echo "• Providing access to interactive timeline visualization"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./run_web_app.sh              # Parse journals and start web app"
    echo "  ./run_web_app.sh --parse-only # Only parse journals, don't start app"
    echo "  ./run_web_app.sh --dev-only   # Only start web app (skip parsing)"
    echo "  ./run_web_app.sh --help       # Show this help message"
    echo ""
    echo -e "${YELLOW}Features:${NC}"
    echo "• 📊 Interactive D3.js timeline visualization"
    echo "• 📝 Enhanced development journal interface with status tracking"
    echo "• 📈 Evaluation results dashboard with metric cards"
    echo "• 📚 Research documentation with markdown rendering"
    echo "• 🎨 Academic-quality design suitable for presentations"
    echo ""
    echo -e "${YELLOW}Requirements:${NC}"
    echo "• Python 3.x with required packages"
    echo "• Node.js and npm for React development"
    echo "• Development journal files (dev_journal_phase*.md)"
    echo "• Analysis results in results/ directory"
    exit 0
fi

print_header() {
    echo -e "${CYAN}🚀 TIMELINE ANALYSIS WEB APPLICATION LAUNCHER${NC}"
    echo -e "${CYAN}Phase 6 Enhanced Visualization & Development Journal Integration${NC}"
    echo "======================================================================"
    echo ""
}

check_prerequisites() {
    echo -e "${BLUE}🔍 CHECKING PREREQUISITES${NC}"
    echo "========================================"
    
    # Check if we're in the correct directory
    if [ ! -f "${PROJECT_ROOT}/run_timeline_analysis.py" ]; then
        echo -e "${RED}❌ Error: Must run from timeline project root directory${NC}"
        echo "Current directory: ${PROJECT_ROOT}"
        echo "Expected files: run_timeline_analysis.py, run_evaluation.py"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        echo "Please install Python 3.x"
        exit 1
    fi
    echo -e "${GREEN}✅ Python 3 available${NC}"
    
    # Check Node.js and npm (only if not dev-only mode)
    if [ "$PARSE_ONLY" = false ]; then
        if ! command -v node &> /dev/null; then
            echo -e "${RED}❌ Node.js not found${NC}"
            echo "Please install Node.js and npm"
            exit 1
        fi
        echo -e "${GREEN}✅ Node.js available: $(node --version)${NC}"
        
        if ! command -v npm &> /dev/null; then
            echo -e "${RED}❌ npm not found${NC}"
            echo "Please install npm"
            exit 1
        fi
        echo -e "${GREEN}✅ npm available: $(npm --version)${NC}"
    fi
    
    # Check for development journal files
    journal_count=$(ls dev_journal_phase*.md 2>/dev/null | wc -l)
    if [ "$journal_count" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Warning: No development journal files found${NC}"
        echo "Expected files: dev_journal_phase1.md, dev_journal_phase2.md, etc."
    else
        echo -e "${GREEN}✅ Found ${journal_count} development journal files${NC}"
    fi
    
    # Check for web app directory
    if [ ! -d "$WEB_APP_DIR" ]; then
        echo -e "${RED}❌ Web app directory not found: ${WEB_APP_DIR}${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Web app directory exists${NC}"
    
    echo ""
}

parse_development_journals() {
    echo -e "${PURPLE}📝 PROCESSING DEVELOPMENT JOURNALS${NC}"
    echo "========================================"
    echo "Converting journal files to professional HTML format for enhanced web interface..."
    echo ""
    
    # Check if HTML conversion script exists
    HTML_CONVERT_SCRIPT="${PROJECT_ROOT}/scripts/convert_journals_to_pdf.py"
    if [ ! -f "$HTML_CONVERT_SCRIPT" ]; then
        echo -e "${RED}❌ HTML conversion script not found: ${HTML_CONVERT_SCRIPT}${NC}"
        echo "Please ensure scripts/convert_journals_to_pdf.py exists"
        exit 1
    fi
    
    # Run the HTML conversion script
    echo -e "${CYAN}🔄 Converting journals to professional HTML format...${NC}"
    if python3 "$HTML_CONVERT_SCRIPT"; then
        echo ""
        echo -e "${GREEN}✅ Journal HTML conversion completed successfully${NC}"
        
        # Clean up table of contents entries
        HTML_CLEAN_SCRIPT="${PROJECT_ROOT}/scripts/clean_html_toc.py"
        if [ -f "$HTML_CLEAN_SCRIPT" ]; then
            echo -e "${CYAN}🔄 Cleaning table of contents entries...${NC}"
            if python3 "$HTML_CLEAN_SCRIPT"; then
                echo -e "${GREEN}✅ Table of contents cleaning completed successfully${NC}"
            else
                echo -e "${YELLOW}⚠️  Table of contents cleaning failed (non-critical)${NC}"
            fi
        fi
        
        # Check HTML output
        HTML_OUTPUT_DIR="${WEB_APP_DIR}/public/journals"
        if [ -d "$HTML_OUTPUT_DIR" ]; then
            html_count=$(ls "${HTML_OUTPUT_DIR}"/*.html 2>/dev/null | wc -l)
            css_count=$(ls "${HTML_OUTPUT_DIR}"/*.css 2>/dev/null | wc -l)
            echo -e "${GREEN}📁 Generated ${html_count} HTML files and ${css_count} CSS files in: ${HTML_OUTPUT_DIR}${NC}"
        fi
    else
        echo -e "${RED}❌ Journal HTML conversion failed${NC}"
        echo "Check the error messages above and ensure pandoc is installed"
        exit 1
    fi
    
    # Legacy JSON parsing (keeping for backward compatibility if needed)
    if [ -f "$PARSE_SCRIPT" ]; then
        echo -e "${CYAN}🔄 Running legacy JSON preprocessing for compatibility...${NC}"
        if python3 "$PARSE_SCRIPT"; then
            echo -e "${GREEN}✅ Legacy JSON parsing completed${NC}"
            
            # Check JSON output
            if [ -d "$PARSED_OUTPUT_DIR" ]; then
                json_count=$(ls "${PARSED_OUTPUT_DIR}"/*.json 2>/dev/null | wc -l)
                echo -e "${GREEN}📁 Generated ${json_count} JSON files in: ${PARSED_OUTPUT_DIR}${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  Legacy JSON parsing failed, but HTML conversion succeeded${NC}"
        fi
    fi
    
    echo ""
}

setup_web_app() {
    echo -e "${BLUE}🔧 SETTING UP WEB APPLICATION${NC}"
    echo "========================================"
    
    # Navigate to web app directory
    cd "$WEB_APP_DIR"
    
    # Check if node_modules exists and package.json is newer
    if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
        echo -e "${CYAN}📦 Installing/updating npm dependencies...${NC}"
        if npm install; then
            echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
        else
            echo -e "${RED}❌ npm install failed${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Dependencies up to date${NC}"
    fi
    
    # Verify critical dependencies
    echo -e "${CYAN}🔍 Verifying critical dependencies...${NC}"
    if npm list react @vitejs/plugin-react tailwindcss d3 &> /dev/null; then
        echo -e "${GREEN}✅ All critical dependencies available${NC}"
    else
        echo -e "${YELLOW}⚠️  Some dependencies may be missing, but attempting to continue...${NC}"
    fi
    
    echo ""
}

launch_web_app() {
    echo -e "${GREEN}🌐 LAUNCHING WEB APPLICATION${NC}"
    echo "========================================"
    echo "Starting React development server with live timeline visualization..."
    echo ""
    
    echo -e "${CYAN}📊 Available Features:${NC}"
    echo "• Interactive Timeline: D3.js visualization of research paradigm evolution"
    echo "• Development Journals: Enhanced interface with status tracking and phase navigation"
    echo "• Evaluation Results: Metric cards and assessment summaries"
    echo "• Research Documentation: Markdown rendering with syntax highlighting"
    echo ""
    
    echo -e "${YELLOW}🔗 Web application will be available at:${NC}"
    echo "• Primary: http://localhost:5173"
    echo "• Backup: http://localhost:5174 (if 5173 is in use)"
    echo "• Backup: http://localhost:5175 (if 5174 is in use)"
    echo ""
    
    echo -e "${PURPLE}🚀 Starting development server...${NC}"
    echo "Press Ctrl+C to stop the server"
    echo "========================================"
    
    # Start the development server
    # Note: This will run indefinitely until user stops it
    npm run dev
}

main() {
    print_header
    
    # Step 1: Check prerequisites
    check_prerequisites
    
    # Step 2: Parse journals (unless dev-only mode)
    if [ "$DEV_ONLY" = false ]; then
        parse_development_journals
    else
        echo -e "${YELLOW}⏭️  Skipping journal parsing (--dev-only mode)${NC}"
        echo ""
    fi
    
    # Step 3: Setup and launch web app (unless parse-only mode)
    if [ "$PARSE_ONLY" = false ]; then
        setup_web_app
        launch_web_app
    else
        echo -e "${YELLOW}⏭️  Skipping web app launch (--parse-only mode)${NC}"
        echo -e "${GREEN}✅ Journal parsing complete. JSON files ready for web application.${NC}"
    fi
}

# Handle script interruption gracefully
trap 'echo -e "\n${YELLOW}🛑 Web application stopped${NC}"; exit 0' INT

# Run main function
main "$@" 