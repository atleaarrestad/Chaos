import { createBrowserRouter } from 'react-router-dom';
import { lazy, Suspense, type ReactNode } from 'react';
import Layout from '@/components/Layout/Layout';

const Home        = lazy(() => import('@/pages/Home/Home'));
const ComingSoon  = lazy(() => import('@/pages/ComingSoon/ComingSoon'));
const ControlsTest = lazy(() => import('@/pages/ControlsTest/ControlsTest'));
const Lorenz          = lazy(() => import('@/pages/Lorenz/Lorenz'));
const Mandelbrot      = lazy(() => import('@/pages/Mandelbrot/Mandelbrot'));
const DoublePendulum  = lazy(() => import('@/pages/DoublePendulum/DoublePendulum'));
const Cardioid        = lazy(() => import('@/pages/Cardioid/Cardioid'));

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
      { path: 'lorenz', element: <Lazy><Lorenz /></Lazy> },
      { path: 'mandelbrot', element: <Lazy><Mandelbrot /></Lazy> },
      { path: 'cardioid', element: <Lazy><Cardioid /></Lazy> },
      { path: 'bifurcation', element: <Lazy><ComingSoon /></Lazy> },
      { path: 'double-pendulum', element: <Lazy><DoublePendulum /></Lazy> },
      { path: 'controls-test', element: <Lazy><ControlsTest /></Lazy> },
    ],
  },
]);
