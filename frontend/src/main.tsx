import React from 'react';
import ReactDOM from 'react-dom/client';
import {createClient} from '@supabase/supabase-js';
import App from './App';
import {configureApiClient} from './api';
import './style.css';

const root = ReactDOM.createRoot(document.getElementById('root')!);

async function start() {
  const response = await fetch('/api/public-config');
  if (!response.ok) throw new Error('The application configuration could not be loaded.');
  const config = await response.json();
  if (config.local_mode) {
    configureApiClient(null);
    root.render(<React.StrictMode><App supabase={null}/></React.StrictMode>);
    return;
  }
  if (!config.supabase_url || !config.supabase_anon_key) {
    throw new Error('Account sign-in is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY to the server environment.');
  }
  const supabase = createClient(config.supabase_url, config.supabase_anon_key);
  configureApiClient(supabase);
  root.render(<React.StrictMode><App supabase={supabase}/></React.StrictMode>);
}

start().catch(error => {
  const message = error instanceof Error ? error.message : 'The application could not start.';
  root.render(
    <main className="startup-error">
      <section>
        <h1>DocuMind setup required</h1>
        <p>{message}</p>
        <p>For public access, configure the Supabase project URL and public anon key on the server, then reload.</p>
      </section>
    </main>,
  );
});
