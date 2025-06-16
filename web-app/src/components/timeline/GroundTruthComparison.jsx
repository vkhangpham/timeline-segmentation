import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

function GroundTruthComparison({ timelineData, groundTruthData }) {
  const svgRef = useRef()
  const containerRef = useRef()
  const [dimensions, setDimensions] = useState({ width: 800, height: 300 })

  // Color scales
  const predictedColors = ['#a5d8ff', '#b9f6ca', '#ffe57f', '#ffd6a5', '#cfd8dc', '#f8bbd0', '#d7ccc8']
  const groundTruthColors = ['#1e40af', '#059669', '#d97706', '#dc2626', '#7c3aed', '#be185d', '#374151']

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
    if (!timelineData || !groundTruthData || !timelineData.timeline_analysis?.final_period_characterizations) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const { width, height } = dimensions
    const margin = { top: 20, right: 20, bottom: 80, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Get data
    const predictedSegments = timelineData.timeline_analysis.final_period_characterizations
    const groundTruthPeriods = groundTruthData.historical_periods
    const timeRange = [
      Math.min(
        timelineData.analysis_metadata.time_range[0],
        Math.min(...groundTruthPeriods.map(p => p.start_year))
      ),
      Math.max(
        timelineData.analysis_metadata.time_range[1],
        Math.max(...groundTruthPeriods.map(p => p.end_year))
      )
    ]

    console.log('Ground truth comparison:', {
      predictedSegments: predictedSegments.length,
      groundTruthPeriods: groundTruthPeriods.length,
      timeRange
    })

    // Create scales
    const xScale = d3.scaleLinear()
      .domain(timeRange)
      .range([0, innerWidth])

    // Create main group
    const g = svg
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'comparison-svg')
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Create tooltip
    let tooltip = d3.select('body').select('.comparison-tooltip')
    if (tooltip.empty()) {
      tooltip = d3.select('body')
        .append('div')
        .attr('class', 'comparison-tooltip')
        .style('position', 'absolute')
        .style('background', 'white')
        .style('border', '1px solid rgba(0,0,0,0.1)')
        .style('border-radius', '4px')
        .style('padding', '8px 12px')
        .style('font-size', '14px')
        .style('box-shadow', '0 2px 8px rgba(0,0,0,0.15)')
        .style('pointer-events', 'none')
        .style('z-index', '1000')
        .style('opacity', 0)
    }

    // Draw timeline axis
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d3.format('d'))
      .ticks(Math.min(10, timeRange[1] - timeRange[0]))

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

    // Track heights for segments
    const segmentHeight = 40
    const gapBetween = 20

    // Draw predicted segments (top)
    const predictedY = 10
    g.append('text')
      .attr('x', -10)
      .attr('y', predictedY + segmentHeight / 2)
      .attr('text-anchor', 'end')
      .attr('fill', '#666')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('Predicted')

    const predictedBars = g.selectAll('.predicted-bar')
      .data(predictedSegments)
      .join('rect')
      .attr('class', 'predicted-bar')
      .attr('x', d => xScale(d.period[0]))
      .attr('y', predictedY)
      .attr('width', d => Math.max(1, xScale(d.period[1]) - xScale(d.period[0])))
      .attr('height', segmentHeight)
      .attr('fill', (d, i) => predictedColors[i % predictedColors.length])
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .style('opacity', 0.8)

    // Draw ground truth periods (bottom)
    const groundTruthY = predictedY + segmentHeight + gapBetween
    g.append('text')
      .attr('x', -10)
      .attr('y', groundTruthY + segmentHeight / 2)
      .attr('text-anchor', 'end')
      .attr('fill', '#666')
      .style('font-size', '12px')
      .style('font-weight', '600')
      .text('Ground Truth')

    const groundTruthBars = g.selectAll('.groundtruth-bar')
      .data(groundTruthPeriods)
      .join('rect')
      .attr('class', 'groundtruth-bar')
      .attr('x', d => xScale(d.start_year))
      .attr('y', groundTruthY)
      .attr('width', d => Math.max(1, xScale(d.end_year) - xScale(d.start_year)))
      .attr('height', segmentHeight)
      .attr('fill', (d, i) => groundTruthColors[i % groundTruthColors.length])
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .style('opacity', 0.8)

    // Add interactivity for predicted segments
    predictedBars
      .on('mouseenter', function(event, d) {
        d3.select(this).style('opacity', 1)
        tooltip
          .style('opacity', 1)
          .html(`
            <div><strong>Predicted: ${d.topic_label}</strong></div>
            <div>Period: ${d.period[0]} - ${d.period[1]}</div>
            <div>Confidence: ${(d.change_confidence * 100).toFixed(1)}%</div>
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseleave', function() {
        d3.select(this).style('opacity', 0.8)
        tooltip.style('opacity', 0)
      })

    // Add interactivity for ground truth periods
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

    // Add connecting lines for overlapping periods
    const connections = []
    predictedSegments.forEach((predicted, pIndex) => {
      groundTruthPeriods.forEach((truth, tIndex) => {
        const overlapStart = Math.max(predicted.period[0], truth.start_year)
        const overlapEnd = Math.min(predicted.period[1], truth.end_year)
        if (overlapStart < overlapEnd) {
          const overlapRatio = (overlapEnd - overlapStart) / (predicted.period[1] - predicted.period[0])
          if (overlapRatio > 0.3) { // Only show significant overlaps
            connections.push({
              predicted: pIndex,
              groundTruth: tIndex,
              overlapRatio,
              overlapStart,
              overlapEnd
            })
          }
        }
      })
    })

    // Draw connection lines
    connections.forEach(conn => {
      const predictedCenter = (xScale(predictedSegments[conn.predicted].period[0]) + 
                              xScale(predictedSegments[conn.predicted].period[1])) / 2
      const groundTruthCenter = (xScale(groundTruthPeriods[conn.groundTruth].start_year) + 
                                xScale(groundTruthPeriods[conn.groundTruth].end_year)) / 2
      
      g.append('line')
        .attr('x1', predictedCenter)
        .attr('y1', predictedY + segmentHeight)
        .attr('x2', groundTruthCenter)
        .attr('y2', groundTruthY)
        .attr('stroke', '#666')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '2,2')
        .style('opacity', 0.6)
    })

  }, [timelineData, groundTruthData, dimensions])

  if (!timelineData || !groundTruthData) {
    return null
  }

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Predicted vs Ground Truth Comparison</h3>
          <p className="text-sm text-gray-600">
            Comparing algorithm-detected segments with research-backed ground truth periods
          </p>
        </div>
        <div ref={containerRef} style={{ border: '1px solid #ccc', minHeight: '300px' }}>
          <svg ref={svgRef} style={{ display: 'block', width: '100%', height: '300px' }}></svg>
        </div>
      </div>
    </div>
  )
}

export default GroundTruthComparison 