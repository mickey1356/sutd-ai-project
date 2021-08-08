import React from 'react';
import { Link } from 'react-router-dom';
import StackedBarChart from '../../charts/StackedBarChart';
import Icon from '../../images/icon-01.svg';
import EditMenu from '../EditMenu';
import Image01 from '../../images/wsb_kid.png';

// Import utilities
import { tailwindConfig, hexToRGB } from '../../utils/Utils';

function DashboardCard01({firebaseData, date}) {
  if (!date.selectedDate || !firebaseData[date.selectedDate]){
    return null
  }
  const dayData = firebaseData[date.selectedDate]
  const chartData = {
    labels: ['BB', 'AMC', 'NOK', 'GME'],
    datasets: [
      // Indigo line
      {
        label: 'Number of Negative Sentiments',
        data: [dayData["BB"].negative_count, dayData["AMC"].negative_count, dayData["NOK"].negative_count, dayData["GME"].negative_count],
        fill: true,
        backgroundColor: `rgba(${hexToRGB(tailwindConfig().theme.colors.red[500])}, 0.08)`,
        borderColor: tailwindConfig().theme.colors.red[500],
        borderWidth: 2,
        tension: 0,
        pointRadius: 0,
        pointHoverRadius: 3,
        pointBackgroundColor: tailwindConfig().theme.colors.red[500],
        clip: 20,
        order: 1, 
      },
      //
      {
        label: 'Number of Positive Sentiments',
        data: [dayData["BB"].positive_count, dayData["AMC"].positive_count, dayData["NOK"].positive_count, dayData["GME"].positive_count],
        fill: true,
        backgroundColor: `rgba(${hexToRGB(tailwindConfig().theme.colors.green[500])}, 0.08)`,
        borderColor: tailwindConfig().theme.colors.green[500],
        borderWidth: 2,
        tension: 0,
        pointRadius: 0,
        pointHoverRadius: 3,
        pointBackgroundColor: tailwindConfig().theme.colors.green[500],
        clip: 20,
        order: 1, 
      },
      // Gray line
      // {
      //   label: 'Total mentions',
      //   data: [dayData["BB"].positive_count + dayData["BB"].neutral_count + dayData["BB"].negative_count, 
      //   dayData["AMC"].positive_count + dayData["AMC"].neutral_count + dayData["AMC"].negative_count, 
      //   dayData["NOK"].positive_count + dayData["NOK"].neutral_count + dayData["NOK"].negative_count,
      //   dayData["GME"].positive_count + dayData["GME"].neutral_count + dayData["GME"].negative_count],
      //   borderColor: tailwindConfig().theme.colors.blue[300],
      //   borderWidth: 2,
      //   tension: 0,
      //   pointRadius: 0,
      //   pointHoverRadius: 3,
      //   pointBackgroundColor: tailwindConfig().theme.colors.blue[300],
      //   clip: 20,
      //   type:'line',
      //   order: 0,
      // },
    ],
  };

  return (
    <div className="flex flex-col col-span-full bg-grayshadow-lg rounded-sm border border-gray-200 justify-center">
      <div className="px-5 pt-5">
        <header className="flex justify-between items-start mb-2">
        
          {/* Icon */}
          {/* <img src={Icon} width="32" height="32" alt="Icon 01" /> */}
          {/* Menu button */}
          {/* <EditMenu className="relative inline-flex">
            <li>
              <Link className="font-medium text-sm text-gray-600 hover:text-gray-200 flex py-1 px-3" to="#0">Option 1</Link>
            </li>
            <li>
              <Link className="font-medium text-sm text-gray-600 hover:text-gray-200 flex py-1 px-3" to="#0">Option 2</Link>
            </li>
            <li>
              <Link className="font-medium text-sm text-red-500 hover:text-red-600 flex py-1 px-3" to="#0">Remove</Link>
            </li>
          </EditMenu> */}
        </header>
        <h2 className="text-lg font-semibold text-gray-200 mb-2">WallStreetBets Ticker Sentiment</h2>
        {/* <div className="text-xs font-semibold text-gray-400 uppercase mb-1">Sales</div>
        <div className="flex items-start">
          <div className="text-3xl font-bold text-gray-200 mr-2">$24,780</div>
          <div className="text-sm font-semibold text-white px-1.5 bg-green-500 rounded-full">+49%</div>
        </div> */}
      </div>
      {/* Chart built with Chart.js 3 */}
      <div className="flex justify-center"><img className="background-overlay " src={Image01}></img></div>
      <div className="flex-grow">
        {/* Change the height attribute to adjust the chart height */}
        <StackedBarChart data={chartData} width={389} height={500} />
      </div>
    </div>
  );
}

export default DashboardCard01;
