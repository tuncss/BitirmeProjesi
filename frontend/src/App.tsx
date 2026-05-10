import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout/Layout'
import HomePage from './pages/HomePage'
import SegmentationPage from './pages/SegmentationPage'
import ResultsPage from './pages/ResultsPage'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/segment" element={<SegmentationPage />} />
          <Route path="/results/:taskId" element={<ResultsPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
