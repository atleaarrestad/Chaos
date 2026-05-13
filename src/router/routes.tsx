import { createBrowserRouter } from 'react-router-dom';
import { lazy, Suspense, type ReactNode } from 'react';
import Layout from '@/components/Layout/Layout';

const Home        = lazy(() => import('@/pages/Home/Home'));
const ComingSoon  = lazy(() => import('@/pages/ComingSoon/ComingSoon'));
const ControlsTest = lazy(() => import('@/pages/ControlsTest/ControlsTest'));
const Mandelbrot  = lazy(() => import('@/pages/Mandelbrot/Mandelbrot'));

function Lazy({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<div style={{ padding: '2rem', color: 'var(--text-muted)' }}>Loading…</div>}>
      {children}
    </Suspense>
  );
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Lazy><Home /></Lazy> },
      { path: 'lorenz', element: <Lazy><ComingSoon /></Lazy> },
      { path: 'mandelbrot', element: <Lazy><Mandelbrot /></Lazy> },
      { path: 'julia', element: <Lazy><ComingSoon /></Lazy> },
      { path: 'cardioid', element: <Lazy><ComingSoon /></Lazy> },
      { path: 'bifurcation', element: <Lazy><ComingSoon /></Lazy> },
      { path: 'double-pendulum', element: <Lazy><ComingSoon /></Lazy> },
      { path: 'controls-test', element: <Lazy><ControlsTest /></Lazy> },
    ],
  },
]);
