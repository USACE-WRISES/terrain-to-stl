import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';

const ConvertPage = lazy(async () => {
  const module = await import('./pages/ConvertPage');
  return { default: module.ConvertPage };
});

const ViewerPage = lazy(async () => {
  const module = await import('./pages/ViewerPage');
  return { default: module.ViewerPage };
});

export default function App() {
  return (
    <div className="app-shell">
      <main className="app-main">
        <Suspense fallback={<div className="panel loading-panel">Loading page...</div>}>
          <Routes>
            <Route path="/" element={<ViewerPage />} />
            <Route path="/viewer" element={<ViewerPage />} />
            <Route path="/convert" element={<ConvertPage />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
