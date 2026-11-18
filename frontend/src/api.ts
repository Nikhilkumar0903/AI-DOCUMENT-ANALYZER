import type {SupabaseClient} from '@supabase/supabase-js';

let client: SupabaseClient | null = null;

export function configureApiClient(supabase: SupabaseClient | null) {
  client = supabase;
}

export async function api(path: string, init?: RequestInit) {
  const headers = new Headers(init?.headers);
  if (client) {
    const {data, error} = await client.auth.getSession();
    if (error) throw error;
    if (data.session?.access_token) {
      headers.set('Authorization', `Bearer ${data.session.access_token}`);
    }
  }
  return fetch(`/api${path}`, {...init, headers});
}
