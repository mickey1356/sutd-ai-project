import React from 'react';


function HeaderCard({ firebaseData, date }) {
    if (!date.selectedDate || !firebaseData[date.selectedDate]) {
        return (
            <div className="flex col-span-full justify-center">
                <h1 className="text-sm font-bold text-gray-100 mb-3">Non-trading day</h1>
            </div>)
    }
    const dayData = firebaseData[date.selectedDate]

    return (
        <div className="flex flex-col col-span-full">
            <link rel="stylesheet" href="https://pro.fontawesome.com/releases/v5.10.0/css/all.css" />
            <h2 className="text-lg font-semibold text-gray-200 mb-3">Your Next Best Bet</h2>
            <div className="flex flex-wrap col-span-full bg-grayshadow-lg rounded-sm justify-between">

                <div className="w-full lg:w-80 xl:w-80 px-4">
                    <div className="relative flex flex-col min-w-0 break-words bg-indigo-900 rounded mb-6 xl:mb-0 shadow-lg">
                        <div className="flex-auto p-4">
                            <div className="flex flex-wrap">
                                <div className="relative w-full pr-4 max-w-full flex-grow flex-1">
                                    <h5 className="text-blueGray-400 uppercase font-bold text-xs">
                                        {dayData["BB"].positive_count + dayData["BB"].neutral_count + dayData["BB"].negative_count} mentions (<span className="text-green-500">{dayData["BB"].positive_count}+</span>, <span className="text-red-500">{dayData["BB"].negative_count}-</span>)
                                    </h5>
                                    <div className="mt-1 font-semibold text-2xl text-blueGray-700">
                                        <img src="/BB.jpg" style={{ borderRadius: "50%", display: 'inline-block', marginRight: '0.5rem', verticalAlign: 'middle' }} width="40" height="40"></img>
                                        BB
                                    </div>
                                </div>
                                <div className="relative w-auto pl-4 flex-initial">
                                    {dayData["BB"].pc >= 0.05 ?
                                        (<div className="text-sm font-semibold text-white text-center px-1.5 bg-green-500 rounded-full w-14">BUY</div>) :
                                        (dayData["BB"].pc <= -0.05 ?
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-red-500 rounded-full w-14">SELL</div>) :
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-blue-500 rounded-full w-14">HOLD</div>))
                                    }
                                </div>
                            </div>
                            <p className="text-sm text-blueGray-400 mt-4">

                                {dayData["BB"].pc >= 0 ?
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-green-500"> <i className="fas fa-arrow-up"></i> {(dayData["BB"].pc * 100).toFixed(2)}% </span>potential upside</span>) :
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-red-500"> <i className="fas fa-arrow-down"></i> {(dayData["BB"].pc * 100).toFixed(2)}% </span>potential downside </span>)
                                }
                            </p>
                        </div>
                    </div>
                </div>
                <div className="w-full lg:w-80 xl:w-80 px-4">
                    <div className="relative flex flex-col min-w-0 break-words bg-indigo-900 rounded mb-6 xl:mb-0 shadow-lg">
                        <div className="flex-auto p-4">
                            <div className="flex flex-wrap">
                                <div className="relative w-full pr-4 max-w-full flex-grow flex-1">
                                    <h5 className="text-blueGray-400 uppercase font-bold text-xs">
                                        {dayData["AMC"].positive_count + dayData["AMC"].neutral_count + dayData["AMC"].negative_count} mentions (<span className="text-green-500">{dayData["AMC"].positive_count}+</span>, <span className="text-red-500">{dayData["AMC"].negative_count}-</span>)
                                    </h5>
                                    <div className="mt-1 font-semibold text-2xl text-blueGray-700">
                                        <img src="/AMC.jpg" style={{ borderRadius: "50%", display: 'inline-block', marginRight: '0.5rem', verticalAlign: 'middle' }} width="40" height="40"></img>
                                        AMC
                                    </div>
                                </div>
                                <div className="relative w-auto pl-4 flex-initial">
                                {dayData["AMC"].pc >= 0.05 ?
                                        (<div className="text-sm font-semibold text-white text-center px-1.5 bg-green-500 rounded-full w-14">BUY</div>) :
                                        (dayData["AMC"].pc <= -0.05 ?
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-red-500 rounded-full w-14">SELL</div>) :
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-blue-500 rounded-full w-14">HOLD</div>))
                                    }
                                </div>
                            </div>
                            <p className="text-sm text-blueGray-400 mt-4">

                                {dayData["AMC"].pc >= 0 ?
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-green-500"> <i className="fas fa-arrow-up"></i> {(dayData["AMC"].pc * 100).toFixed(2)}% </span>potential upside</span>) :
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-red-500"> <i className="fas fa-arrow-down"></i> {(dayData["AMC"].pc * 100).toFixed(2)}% </span>potential downside </span>)
                                }
                            </p>
                        </div>
                    </div>
                </div>
                <div className="w-full lg:w-80 xl:w-80 px-4">
                    <div className="relative flex flex-col min-w-0 break-words bg-indigo-900 rounded mb-6 xl:mb-0 shadow-lg">
                        <div className="flex-auto p-4">
                            <div className="flex flex-wrap">
                                <div className="relative w-full pr-4 max-w-full flex-grow flex-1">
                                    <h5 className="text-blueGray-400 uppercase font-bold text-xs">
                                        {dayData["NOK"].positive_count + dayData["NOK"].neutral_count + dayData["NOK"].negative_count} mentions (<span className="text-green-500">{dayData["NOK"].positive_count}+</span>, <span className="text-red-500">{dayData["NOK"].negative_count}-</span>)
                                    </h5>
                                    <div className="mt-1 font-semibold text-2xl text-blueGray-700">
                                        <img src="/NOK.jpg" style={{ borderRadius: "50%", display: 'inline-block', marginRight: '0.5rem', verticalAlign: 'middle' }} width="40" height="40"></img>
                                        NOK
                                    </div>
                                </div>
                                <div className="relative w-auto pl-4 flex-initial">
                                {dayData["NOK"].pc >= 0.05 ?
                                        (<div className="text-sm font-semibold text-white text-center px-1.5 bg-green-500 rounded-full w-14">BUY</div>) :
                                        (dayData["NOK"].pc <= -0.05 ?
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-red-500 rounded-full w-14">SELL</div>) :
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-blue-500 rounded-full w-14">HOLD</div>))
                                    }
                                </div>
                            </div>
                            <p className="text-sm text-blueGray-400 mt-4">

                                {dayData["NOK"].pc >= 0 ?
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-green-500"> <i className="fas fa-arrow-up"></i> {(dayData["NOK"].pc * 100).toFixed(2)}% </span>potential upside</span>) :
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-red-500"> <i className="fas fa-arrow-down"></i> {(dayData["NOK"].pc * 100).toFixed(2)}% </span>potential downside </span>)
                                }
                            </p>
                        </div>
                    </div>
                </div>
                <div className="w-full lg:w-80 xl:w-80 px-4">
                    <div className="relative flex flex-col min-w-0 break-words bg-indigo-900 rounded mb-6 xl:mb-0 shadow-lg">
                        <div className="flex-auto p-4">
                            <div className="flex flex-wrap">
                                <div className="relative w-full pr-4 max-w-full flex-grow flex-1">
                                    <h5 className="text-blueGray-400 uppercase font-bold text-xs">
                                        {dayData["GME"].positive_count + dayData["GME"].neutral_count + dayData["GME"].negative_count} mentions (<span className="text-green-500">{dayData["GME"].positive_count}+</span>, <span className="text-red-500">{dayData["GME"].negative_count}-</span>)
                                    </h5>
                                    <div className="mt-1 font-semibold text-2xl text-blueGray-700">
                                        <img src="/GME.jpg" style={{ borderRadius: "50%", display: 'inline-block', marginRight: '0.5rem', verticalAlign: 'middle' }} width="40" height="40"></img>
                                        GME
                                    </div>
                                </div>
                                <div className="relative w-auto pl-4 flex-initial">
                                {dayData["GME"].pc >= 0.05 ?
                                        (<div className="text-sm font-semibold text-white text-center px-1.5 bg-green-500 rounded-full w-14">BUY</div>) :
                                        (dayData["GME"].pc <= -0.05 ?
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-red-500 rounded-full w-14">SELL</div>) :
                                            (<div className="text-sm font-semibold text-white text-center px-1.5 bg-blue-500 rounded-full w-14">HOLD</div>))
                                    }
                                </div>
                            </div>
                            <p className="text-sm text-blueGray-400 mt-4">

                                {dayData["GME"].pc >= 0 ?
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-green-500"> <i className="fas fa-arrow-up"></i> {(dayData["GME"].pc * 100).toFixed(2)}% </span>potential upside</span>) :
                                    (<span className="text-emerald-500 mr-2 whitespace-nowrap"> <span className="text-sm text-red-500"> <i className="fas fa-arrow-down"></i> {(dayData["GME"].pc * 100).toFixed(2)}% </span>potential downside </span>)
                                }
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    );
}

export default HeaderCard;
