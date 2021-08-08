import React, { useRef, useEffect } from 'react';
import Chart from 'chart.js/auto';
import 'chartjs-adapter-moment';

// Import utilities
import { tailwindConfig, formatValue } from '../utils/Utils';

function StackedBarChart({
  data,
  width,
  height
}) {

  const canvas = useRef(null);

  useEffect(() => {
    const ctx = canvas.current;
    // eslint-disable-next-line no-unused-vars
    const chart = new Chart(ctx, {
      type: 'bar',
      data: data,
      options: {
        responsive: true,
        // chartArea: {
        //   backgroundColor: tailwindConfig().theme.colors.gray[50],
        // },
        layout: {
          padding: 20,
        },
        scales: {
          y: {
            stacked: true,
            beginAtZero: true,
            ticks: {
              autoSkip: false
            },
            grid: {
              drawOnChartArea: false,
              borderColor: 'white'
            },
            
          },
          x: {
            stacked: true,
            // type: 'time',
            // time: {
            //   parser: 'MM-DD-YYYY',
            //   unit: 'month',
            // },
            grid: {
              drawOnChartArea: false,
              borderColor: 'white'
            },
          },
        },
        plugins: {
          tooltip: {
            callbacks: {
              title: () => false, // Disable tooltip title
              // label: (context) => formatValue(context.parsed.y),
            },
          },
          legend: {
            position: 'top',
            display: true,
          },
        },
        interaction: {
          intersect: false,
          mode: 'nearest',
        },
        maintainAspectRatio: false,
      },
    });
    return () => chart.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  return (
    <canvas ref={canvas} width={width} height={height}></canvas>
  );
}

export default StackedBarChart;