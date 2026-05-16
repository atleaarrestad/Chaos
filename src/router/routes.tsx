import { createBrowserRouter } from 'react-router-dom';
import { lazy, Suspense, type ReactNode } from 'react';
import Layout from '@/components/Layout/Layout';

const Home        = lazy(() => import('@/pages/Home/Home'));
const Lorenz          = lazy(() => import('@/pages/Lorenz/Lorenz'));
const Mandelbrot      = lazy(() => import('@/pages/Mandelbrot/Mandelbrot'));
const DoublePendulum  = lazy(() => import('@/pages/DoublePendulum/DoublePendulum'));
const Cardioid        = lazy(() => import('@/pages/Cardioid/Cardioid'));
const Bifurcation     = lazy(() => import('@/pages/Bifurcation/Bifurcation'));
const Koch            = lazy(() => import('@/pages/Koch/Koch'));
const Conway          = lazy(() => import('@/pages/Conway/Conway'));
const ThreeBody       = lazy(() => import('@/pages/ThreeBody/ThreeBody'));
const NotFound        = lazy(() => import('@/pages/NotFound/NotFound'));

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
      { path: 'bifurcation', element: <Lazy><Bifurcation /></Lazy> },
      { path: 'double-pendulum', element: <Lazy><DoublePendulum /></Lazy> },
      { path: 'koch', element: <Lazy><Koch /></Lazy> },
      { path: 'conway', element: <Lazy><Conway /></Lazy> },
      { path: 'three-body', element: <Lazy><ThreeBody /></Lazy> },
      { path: '*', element: <Lazy><NotFound /></Lazy> },
    ],
  },
]);

