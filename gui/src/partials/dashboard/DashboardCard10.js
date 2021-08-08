import React from 'react';

function DashboardCard10({firebaseData, date}) {
  if (!date.selectedDate || !firebaseData[date.selectedDate]){
    return null
  }
  const dayData = firebaseData[date.selectedDate]

  const sentiments = [
    {
      id: '1',
      sentiment: dayData['sentiments'].pos_sen1,
    },
    {
      id: '2',
      sentiment: dayData['sentiments'].pos_sen2,
    },
    {
      id: '3',
      sentiment: dayData['sentiments'].pos_sen3,
    },
  ];

  return (
    <div className="col-span-full xl:col-span-6 bg-grayshadow-lg rounded-sm border border-gray-200">
      <header className="px-5 py-4 border-b border-gray-100">
        <h2 className="font-semibold text-gray-200">Most Positive Sentiments</h2>
      </header>
      <div className="p-3">

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="table-fixed w-full">
            {/* Table header */}
            <thead className="text-xs font-semibold uppercase text-gray-400 bg-gray-800">
              <tr>
                <th className="p-2 whitespace-nowrap">
                  <div className="font-semibold text-left">Post</div>
                </th>
              </tr>
            </thead>
            {/* Table body */}
            <tbody className="text-sm divide-y divide-gray-100">
              {
                sentiments.map(entry => {
                  return (
                    <tr key={entry.id}>
                      <td className="p-2">
                        <p className="text-left font-medium line-clamp-6 text-green-500">{entry.sentiment}</p>
                      </td>
                    </tr>
                  )
                })
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default DashboardCard10;
