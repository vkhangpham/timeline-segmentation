import { useRef, useEffect, useState } from 'react'
import * as d3 from 'd3'
import DetailsPanel from '../details/DetailsPanel'

// Color scales - defined outside component to avoid recreating on each render
const PREDICTED_COLORS = ['#a5d8ff', '#b9f6ca', '#ffe57f', '#ffd6a5', '#cfd8dc', '#f8bbd0', '#d7ccc8']
const GROUND_TRUTH_COLORS = ['#1e40af', '#059669', '#d97706', '#dc2626', '#7c3aed', '#be185d', '#374151']

function TimelineContainer({ timelineData, groundTruthData }) {
  const svgRef = useRef()
  const containerRef = useRef()
  const [selectedSegment, setSelectedSegment] = useState(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 300 })

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const { width } = containerRef.current.getBoundingClientRect()
        setDimensions({ width: Math.max(width - 40, 600), height: 300 })
      }
    }

    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    if (!timelineData || !timelineData.timeline_analysis?.final_period_characterizations) {
      console.log('Timeline data missing or invalid:', timelineData)
      return
    }

    console.log('Rendering timeline with data:', {
      timeRange: timelineData.analysis_metadata?.time_range,
      segments: timelineData.timeline_analysis.final_period_characterizations.length,
      hasGroundTruth: !!groundTruthData
    })

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove() // Clear previous render

    const { width, height } = dimensions
    const margin = { top: 20, right: 20, bottom: 80, left: 100 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Get data
    const segments = timelineData.timeline_analysis.final_period_characterizations
    const groundTruthPeriods = groundTruthData?.historical_periods || []
    
    // Calculate time ranges for both predicted and ground truth
    let predictedTimeRange = timelineData.analysis_metadata?.time_range
    let groundTruthTimeRange = null
    
    if (groundTruthPeriods.length > 0) {
      const gtStart = Math.min(...groundTruthPeriods.map(p => p.start_year))
      const gtEnd = Math.max(...groundTruthPeriods.map(p => p.end_year))
      groundTruthTimeRange = [gtStart, gtEnd]
    }
    
    // Calculate overall time range and detect mismatch
    let overallTimeRange = predictedTimeRange
    let hasMismatch = false
    
    if (groundTruthTimeRange) {
      overallTimeRange = [
        Math.min(predictedTimeRange[0], groundTruthTimeRange[0]),
        Math.max(predictedTimeRange[1], groundTruthTimeRange[1])
      ]
      
      // Detect significant mismatch (less than 50% overlap)
      const predictedSpan = predictedTimeRange[1] - predictedTimeRange[0]
      const groundTruthSpan = groundTruthTimeRange[1] - groundTruthTimeRange[0]
      const overallSpan = overallTimeRange[1] - overallTimeRange[0]
      
      const predictedCoverage = predictedSpan / overallSpan
      const groundTruthCoverage = groundTruthSpan / overallSpan
      
      // Check for temporal misalignment
      const predictedCenter = (predictedTimeRange[0] + predictedTimeRange[1]) / 2
      const groundTruthCenter = (groundTruthTimeRange[0] + groundTruthTimeRange[1]) / 2
      const centerDistance = Math.abs(predictedCenter - groundTruthCenter)
      
      hasMismatch = predictedCoverage < 0.5 || groundTruthCoverage < 0.5 || centerDistance > overallSpan * 0.3
      
      console.log('Timeline mismatch analysis:', {
        predictedRange: predictedTimeRange,
        groundTruthRange: groundTruthTimeRange,
        overallRange: overallTimeRange,
        predictedCoverage: predictedCoverage.toFixed(2),
        groundTruthCoverage: groundTruthCoverage.toFixed(2),
        centerDistance: centerDistance.toFixed(0),
        hasMismatch
      })
    }
    
    if (!overallTimeRange || overallTimeRange.length !== 2) {
      console.error('Invalid time range:', overallTimeRange)
      return
    }
    
    if (!segments || segments.length === 0) {
      console.error('No segments found:', segments)
      return
    }

    console.log('Timeline setup:', { 
      timeRange: overallTimeRange, 
      segmentCount: segments.length, 
      groundTruthCount: groundTruthPeriods.length, 
      hasMismatch,
      innerWidth, 
      innerHeight 
    })

    // Dynamic label shortening function
    const createLabel = (text, availableWidth) => {
      const charsPerPixel = 0.15
      const maxChars = Math.floor(availableWidth * charsPerPixel)
      
      if (availableWidth < 30) {
        return '' // No label for very small segments
      }
      
      if (text.length <= maxChars) {
        return text // Full text fits
      }
      
      // Try different shortening strategies
      const words = text.split(' ')
      
      if (availableWidth >= 100) {
        // For large segments, try to keep meaningful words
        if (words.length > 1) {
          // Try to keep the most important words (usually first few)
          let shortened = words[0]
          for (let i = 1; i < words.length && shortened.length + words[i].length + 1 <= maxChars - 3; i++) {
            shortened += ' ' + words[i]
          }
          return shortened.length < text.length ? shortened + '...' : shortened
        } else {
          // Single word - truncate if too long
          return maxChars > 3 ? text.substring(0, maxChars - 3) + '...' : text.substring(0, maxChars)
        }
      } else if (availableWidth >= 60) {
        // For medium segments, use first word or truncate
        const firstWord = words[0]
        if (firstWord.length <= maxChars) {
          return firstWord
        } else {
          return maxChars > 3 ? firstWord.substring(0, maxChars - 3) + '...' : firstWord.substring(0, maxChars)
        }
      } else {
        // For small segments, use initials or very short truncation
        if (words.length > 1 && maxChars >= 4) {
          // Create initials from first letters
          const initials = words.map(word => word.charAt(0).toUpperCase()).join('')
          if (initials.length <= maxChars) {
            return initials
          }
        }
        // Fall back to truncation
        return maxChars > 3 ? text.substring(0, maxChars - 3) + '...' : text.substring(0, maxChars)
      }
    }

    // Detect and resolve overlaps using intelligent positioning
    const resolveOverlaps = (labels, isGroundTruth = false) => {
      const resolved = [...labels]
      const minSpacing = 8 // Minimum pixels between labels
      
      // Sort by x position
      resolved.sort((a, b) => a.x - b.x)
      
      // Multiple passes to resolve overlaps
      for (let pass = 0; pass < 3; pass++) {
        let hasOverlap = false
        
        for (let i = 0; i < resolved.length - 1; i++) {
          const current = resolved[i]
          const next = resolved[i + 1]
          
          const currentRight = current.x + (current.textWidth / 2)
          const nextLeft = next.x - (next.textWidth / 2)
          
          if (currentRight + minSpacing > nextLeft) {
            // Overlap detected - resolve by moving labels apart
            hasOverlap = true
            const overlap = (currentRight + minSpacing) - nextLeft
            
            // Move labels apart by half the overlap each
            const adjustment = overlap / 2
            
            // Get segment boundaries based on data structure
            let currentSegmentLeft, currentSegmentRight, nextSegmentLeft, nextSegmentRight
            
            if (isGroundTruth) {
              currentSegmentLeft = xScale(current.data.start_year)
              currentSegmentRight = xScale(current.data.end_year)
              nextSegmentLeft = xScale(next.data.start_year)
              nextSegmentRight = xScale(next.data.end_year)
            } else {
              currentSegmentLeft = xScale(current.data.period[0])
              currentSegmentRight = xScale(current.data.period[1])
              nextSegmentLeft = xScale(next.data.period[0])
              nextSegmentRight = xScale(next.data.period[1])
            }
            
            // Check if we can move current left without going outside its segment
            const newCurrentX = Math.max(
              currentSegmentLeft + (current.textWidth / 2) + 5,
              Math.min(currentSegmentRight - (current.textWidth / 2) - 5, current.x - adjustment)
            )
            
            // Check if we can move next right without going outside its segment
            const newNextX = Math.max(
              nextSegmentLeft + (next.textWidth / 2) + 5,
              Math.min(nextSegmentRight - (next.textWidth / 2) - 5, next.x + adjustment)
            )
            
            current.x = newCurrentX
            next.x = newNextX
          }
        }
        
        if (!hasOverlap) break
      }
      
      return resolved
    }

    // Create scales
    const xScale = d3.scaleLinear()
      .domain(overallTimeRange)
      .range([0, innerWidth])

    // Create main group
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'timeline-svg')
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Add mismatch warning if needed
    if (hasMismatch && groundTruthPeriods.length > 0) {
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('fill', '#dc2626')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('⚠️ Significant temporal mismatch detected')
    }

    // Create tooltip
    let tooltip = d3.select('body').select('.timeline-tooltip')
    if (tooltip.empty()) {
      tooltip = d3.select('body')
        .append('div')
        .attr('class', 'timeline-tooltip')
        .style('position', 'absolute')
        .style('background', 'white')
        .style('border', '1px solid rgba(0,0,0,0.1)')
        .style('border-radius', '4px')
        .style('padding', '8px 12px')
        .style('font-size', '14px')
        .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
        .style('pointer-events', 'none')
        .style('z-index', '1000')
        .style('max-width', '300px')
        .style('opacity', 0)
    }

    // Draw timeline axis
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.format('d'))
      .ticks(Math.min(10, overallTimeRange[1] - overallTimeRange[0]))

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 40)
      .attr('text-anchor', 'middle')
      .attr('fill', '#666')
      .style('font-size', '14px')
      .text('Year')

    // Add vertical grid lines
    const tickValues = xScale.ticks(Math.min(10, overallTimeRange[1] - overallTimeRange[0]))

    // Track heights for segments
    const segmentHeight = 40
    const gapBetween = groundTruthPeriods.length > 0 ? 20 : 0

    // Define positions - Ground Truth on top, Predicted below
    const groundTruthY = 10
    const predictedY = groundTruthPeriods.length > 0 ? (groundTruthY + segmentHeight + gapBetween) : 10

    // Add time range indicators for mismatch
    if (hasMismatch && groundTruthPeriods.length > 0) {
      // Add background shading to show coverage areas
      g.append('rect')
        .attr('x', xScale(groundTruthTimeRange[0]))
        .attr('y', groundTruthY - 5)
        .attr('width', xScale(groundTruthTimeRange[1]) - xScale(groundTruthTimeRange[0]))
        .attr('height', segmentHeight + 10)
        .attr('fill', '#e5f3ff')
        .attr('stroke', '#3b82f6')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .style('opacity', 0.3)
      
      g.append('rect')
        .attr('x', xScale(predictedTimeRange[0]))
        .attr('y', predictedY - 5)
        .attr('width', xScale(predictedTimeRange[1]) - xScale(predictedTimeRange[0]))
        .attr('height', segmentHeight + 10)
        .attr('fill', '#fef3e2')
        .attr('stroke', '#f59e0b')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .style('opacity', 0.3)
    }

    // Add labels if ground truth is available
    if (groundTruthPeriods.length > 0) {
      g.append('text')
        .attr('x', -10)
        .attr('y', groundTruthY + segmentHeight / 2)
        .attr('text-anchor', 'end')
        .attr('fill', '#666')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Ground Truth')
      
      // Add time range indicator for ground truth
      if (hasMismatch) {
        g.append('text')
          .attr('x', -10)
          .attr('y', groundTruthY - 8)
          .attr('text-anchor', 'end')
          .attr('fill', '#3b82f6')
          .style('font-size', '10px')
          .text(`${groundTruthTimeRange[0]}-${groundTruthTimeRange[1]}`)
      }
        
      g.append('text')
        .attr('x', -10)
        .attr('y', predictedY + segmentHeight / 2)
        .attr('text-anchor', 'end')
        .attr('fill', '#666')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Predicted')
      
      // Add time range indicator for predicted
      if (hasMismatch) {
        g.append('text')
          .attr('x', -10)
          .attr('y', predictedY - 8)
          .attr('text-anchor', 'end')
          .attr('fill', '#f59e0b')
          .style('font-size', '10px')
          .text(`${predictedTimeRange[0]}-${predictedTimeRange[1]}`)
      }
    }

    console.log('Drawing segments:', { segmentHeight, gapBetween, segments: segments.length })
    
    // Draw ground truth periods first (top position)
    if (groundTruthPeriods.length > 0) {
      const groundTruthBars = g.selectAll('.groundtruth-bar')
        .data(groundTruthPeriods)
        .join('rect')
        .attr('class', 'groundtruth-bar')
        .attr('x', d => xScale(d.start_year))
        .attr('y', groundTruthY)
        .attr('width', d => Math.max(1, xScale(d.end_year) - xScale(d.start_year)))
        .attr('height', segmentHeight)
        .attr('fill', (d, i) => GROUND_TRUTH_COLORS[i % GROUND_TRUTH_COLORS.length])
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .style('opacity', 0.8)

      // Add ground truth interactivity
      groundTruthBars
        .on('mouseenter', function(event, d) {
          d3.select(this).style('opacity', 1)
          tooltip
            .style('opacity', 1)
            .html(`
              <div><strong>Ground Truth: ${d.period_name}</strong></div>
              <div>Period: ${d.start_year} - ${d.end_year}</div>
              <div>Duration: ${d.duration_years} years</div>
              <div>Confidence: ${(d.confidence * 100).toFixed(0)}%</div>
            `)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
        })
        .on('mouseleave', function() {
          d3.select(this).style('opacity', 0.8)
          tooltip.style('opacity', 0)
        })
        .on('mousemove', function(event) {
          tooltip
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
        })

      // Add ground truth labels with overlap prevention
      const groundTruthLabelData = groundTruthPeriods.map(d => {
        const segmentWidth = xScale(d.end_year) - xScale(d.start_year)
        const centerX = (xScale(d.start_year) + xScale(d.end_year)) / 2
        
        const label = createLabel(d.period_name, segmentWidth)
        
        return {
          data: d,
          label: label,
          x: centerX,
          width: segmentWidth,
          textWidth: label.length * 6.5, // Approximate text width
          shouldShow: label !== '',
          isGroundTruth: true
        }
      }).filter(item => item.shouldShow)

      // Resolve overlaps for ground truth labels
      const resolvedGroundTruthLabels = resolveOverlaps(groundTruthLabelData, true)

      g.selectAll('.groundtruth-label')
        .data(resolvedGroundTruthLabels)
        .join('text')
        .attr('class', 'groundtruth-label')
        .attr('x', d => d.x)
        .attr('y', groundTruthY + segmentHeight / 2 + 5)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('font-weight', '500')
        .style('fill', '#fff')
        .style('pointer-events', 'none')
        .text(d => d.label)
    }

    // Draw predicted segments (bottom position)
    const segmentBars = g.selectAll('.segment-bar')
      .data(segments)
      .join('rect')
      .attr('class', 'segment-bar')
      .attr('x', d => {
        const x = xScale(d.period[0])
        console.log(`Segment ${d.topic_label}: x=${x}, start=${d.period[0]}`)
        return x
      })
      .attr('y', predictedY)
      .attr('width', d => {
        const width = Math.max(1, xScale(d.period[1]) - xScale(d.period[0]))
        console.log(`Segment ${d.topic_label}: width=${width}, period=${d.period}`)
        return width
      })
      .attr('height', segmentHeight)
      .attr('fill', (d, i) => {
        const color = PREDICTED_COLORS[i % PREDICTED_COLORS.length]
        console.log(`Segment ${i}: color=${color}`)
        return color
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .style('opacity', 0.8)

    // Add interactivity
    segmentBars
      .on('mouseenter', function(event, d) {
        d3.select(this).style('opacity', 1)
        
        tooltip
          .style('opacity', 1)
          .html(`
            <div><strong>${d.topic_label}</strong></div>
            <div>Period: ${d.period[0]} - ${d.period[1]}</div>
            <div>Confidence: ${(d.confidence * 100).toFixed(1)}%</div>
            <div>Papers: ${d.representative_papers.length}</div>
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseleave', function() {
        d3.select(this).style('opacity', 0.8)
        tooltip.style('opacity', 0)
      })
      .on('mousemove', function(event) {
        tooltip
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('click', function(event, d) {
        setSelectedSegment(d)
        
        // Visual feedback for selection
        segmentBars.style('opacity', 0.6)
        d3.select(this).style('opacity', 1)
      })

    // Add predicted segment labels with overlap prevention
    const labelData = segments.map(d => {
      const segmentWidth = xScale(d.period[1]) - xScale(d.period[0])
      const centerX = (xScale(d.period[0]) + xScale(d.period[1])) / 2
      
      const label = createLabel(d.topic_label, segmentWidth)
      
      return {
        data: d,
        label: label,
        x: centerX,
        width: segmentWidth,
        textWidth: label.length * 6.5, // Approximate text width
        shouldShow: label !== ''
      }
    }).filter(item => item.shouldShow)

    const resolvedLabels = resolveOverlaps(labelData, false)

    g.selectAll('.segment-label')
      .data(resolvedLabels)
      .join('text')
      .attr('class', 'segment-label')
      .attr('x', d => d.x)
      .attr('y', predictedY + segmentHeight / 2 + 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-weight', '500')
      .style('fill', '#333')
      .style('pointer-events', 'none')
      .text(d => d.label)

    // Add vertical grid lines after segments are positioned
    const timelineTop = groundTruthPeriods.length > 0 ? groundTruthY : predictedY
    const timelineBottom = predictedY + segmentHeight
    
    g.selectAll('.grid-line')
      .data(tickValues)
      .join('line')
      .attr('class', 'grid-line')
      .attr('x1', d => xScale(d))
      .attr('x2', d => xScale(d))
      .attr('y1', timelineTop - 5)
      .attr('y2', timelineBottom + 5)
      .attr('stroke', '#d1d5db')
      .attr('stroke-width', 1)
      .style('opacity', 0.6)
      .style('pointer-events', 'none')

    // Add transition markers for predicted segments
    const transitions = segments.slice(1).map((segment, i) => ({
      year: segment.period[0],
      fromSegment: segments[i],
      toSegment: segment
    }))

    g.selectAll('.transition-marker')
      .data(transitions)
      .join('line')
      .attr('class', 'transition-marker')
      .attr('x1', d => xScale(d.year))
      .attr('x2', d => xScale(d.year))
      .attr('y1', predictedY - 5)
      .attr('y2', predictedY + segmentHeight + 5)
      .attr('stroke', '#333')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4,4')
      .style('opacity', 0.7)

  }, [timelineData, groundTruthData, dimensions])

  if (!timelineData) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No timeline data available
      </div>
    )
  }

  if (!timelineData.timeline_analysis?.final_period_characterizations) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Invalid timeline data structure: missing period_characterizations
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div ref={containerRef} className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            {groundTruthData ? 'Predicted vs Ground Truth Timeline' : 'Research Evolution Timeline'}
          </h3>
          <p className="text-sm text-gray-600">
            {groundTruthData 
              ? 'Compare algorithm-detected segments with research-backed ground truth periods. Click on segments to explore details.'
              : 'Click on a segment to explore research paradigms and representative papers'
            }
          </p>
        </div>
        <div style={{ border: '1px solid #ccc', minHeight: '300px' }}>
          <svg ref={svgRef} style={{ display: 'block', width: '100%', height: '300px' }}></svg>
        </div>
      </div>

      {selectedSegment && (
        <DetailsPanel 
          segment={selectedSegment} 
          onClose={() => setSelectedSegment(null)}
        />
      )}
    </div>
  )
}

export default TimelineContainer 