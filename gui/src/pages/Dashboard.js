import React, { useState, useEffect} from 'react';
import firebase from 'firebase';

import Sidebar from '../partials/Sidebar';
import Header from '../partials/Header';
import WelcomeBanner from '../partials/dashboard/WelcomeBanner';
import FilterButton from '../partials/actions/FilterButton';
import Datepicker from '../partials/actions/Datepicker';
import DashboardCard01 from '../partials/dashboard/DashboardCard01';
import DashboardCard10 from '../partials/dashboard/DashboardCard10';
import DashboardCard13 from '../partials/dashboard/DashboardCard13';
import HeaderCard from '../partials/dashboard/HeaderCard';
import Banner from '../partials/Banner';

const firebaseConfig = {
  apiKey: "<add firebase api key here>",
  authDomain: "<add firebase authentication domain here>",
  databaseUrl: "<add firebase database url here>",
  projectId: "<add firebase project id here>",
  storageBucket: "<add firebase storage bucket here>"
};

if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
} else {
  firebase.app(); // if already initialized, use that one
}

const getFirebaseData = () => {
  return firebase.database().ref('/')
};

function Dashboard() {

  // const [sidebarOpen, setSidebarOpen] = useState(false);
  const [firebaseData, setFirebaseData] = useState();
  const [date, setDate] = useState({selectedDate: null, minDate: null, maxDate: null});

  const onDataChange = (data) => {
    setFirebaseData(data.val());
    const dates = Object.keys(data.val()).sort()
    const minDate = dates[0]
    const maxDate = dates[dates.length-1]
    const date = {
      selectedDate: maxDate,
      minDate: minDate,
      maxDate: maxDate,
    }
    setDate(date)
  };

  useEffect(() => {
    getFirebaseData().on("value", onDataChange);

    return () => {
      getFirebaseData().off("value", onDataChange);
    };
  }, []);



  return (
    <div className="flex h-screen overflow-hidden">

      {/* Sidebar */}
      {/* <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} /> */}

      {/* Content area */}
      <div className="relative flex flex-col flex-1 overflow-y-auto overflow-x-hidden">

        {/*  Site header */}
        {/* <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} /> */}

        <main>
          <div className="px-4 sm:px-6 lg:px-8 py-8 w-full max-w-9xl mx-auto">

            {/* Welcome banner */}
            <WelcomeBanner />

            {/* Dashboard actions */}
            <div className="sm:flex sm:justify-between sm:items-center mb-8">
              <div className="w-64"></div>
              {/* Left: Avatars */}
              {/* <DashboardAvatars /> */}
              <h1 className="text-5xl font-bold text-gray-100 mb-3">Trading Day: {date.selectedDate}</h1>
              {/* Right: Actions */}
              <div className="grid grid-flow-col sm:auto-cols-max justify-start sm:justify-end gap-2">
                
                {/* Filter button */}
                {/* <FilterButton /> */}
                {/* Datepicker built with flatpickr */}
                <Datepicker date={date} setDate={setDate}/>
                {/* Add view button */}
                {/* <button className="btn bg-indigo-500 hover:bg-indigo-600 text-white">
                  <svg className="w-4 h-4 fill-current opacity-50 flex-shrink-0" viewBox="0 0 16 16">
                    
                    <path d="M15 7H9V1c0-.6-.4-1-1-1S7 .4 7 1v6H1c-.6 0-1 .4-1 1s.4 1 1 1h6v6c0 .6.4 1 1 1s1-.4 1-1V9h6c.6 0 1-.4 1-1s-.4-1-1-1z" />
                  </svg>
                  <svg className="w-4 h-4 fill-current text-indigo-50" viewBox="10 10 16 16">
                    <path d="M18 10c-4.4 0-8 3.1-8 7s3.6 7 8 7h.6l5.4 2v-4.4c1.2-1.2 2-2.8 2-4.6 0-3.9-3.6-7-8-7zm4 10.8v2.3L18.9 22H18c-3.3 0-6-2.2-6-5s2.7-5 6-5 6 2.2 6 5c0 2.2-2 3.8-2 3.8z" />
                  </svg>
                  <span className="hidden xs:block ml-2">View Sentiments</span>
                </button> */}
                
              </div>
              

            </div>


            {/* Cards */}
            <div className="grid grid-cols-12 gap-6">

              <HeaderCard firebaseData={firebaseData} date={date}/>

              {/* Line chart (Acme Plus) */}
              <DashboardCard01 firebaseData={firebaseData} date={date}/>
              {/* Card (Positive Sentiments) */}
              <DashboardCard10 firebaseData={firebaseData} date={date}/>
              {/* Card (Negative Sentiments) */}
              <DashboardCard13 firebaseData={firebaseData} date={date}/>
              {/* Line chart (Acme Advanced) */}
              {/* <DashboardCard02 /> */}
              {/* Line chart (Acme Professional) */}
              {/* <DashboardCard03 /> */}
              {/* Bar chart (Direct vs Indirect) */}
              {/* <DashboardCard04 /> */}
              {/* Line chart (Real Time Value) */}
              {/* <DashboardCard05 /> */}
              {/* Doughnut chart (Top Countries) */}
              {/* <DashboardCard06 /> */}
              {/* Table (Top Channels) */}
              {/* <DashboardCard07 /> */}
              {/* Line chart (Sales Over Time) */}
              {/* <DashboardCard08 /> */}
              {/* Stacked bar chart (Sales VS Refunds) */}
              {/* <DashboardCard09 /> */}
              {/* Card (Reasons for Refunds) */}
              {/* <DashboardCard11 /> */}
              {/* Card (Recent Activity) */}
              {/* <DashboardCard12 /> */}

            </div>

          </div>
        </main>

        {/* <Banner /> */}

      </div>
    </div>
  );
}

export default Dashboard;