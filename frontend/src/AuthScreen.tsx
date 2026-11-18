import {useEffect, useState} from 'react';
import type {FormEvent} from 'react';
import type {SupabaseClient} from '@supabase/supabase-js';
import {BrainCircuit, LoaderCircle, Mail, ShieldCheck} from 'lucide-react';
import './auth.css';

type AuthMode = 'sign-in' | 'sign-up' | 'reset' | 'recovery';

type Props = {
  supabase: SupabaseClient;
  mode: AuthMode;
  onModeChange: (mode: AuthMode) => void;
};

export default function AuthScreen({supabase, mode, onModeChange}: Props) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    setError('');
    setMessage('');
  }, [mode]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    setMessage('');
    try {
      if (mode === 'sign-up') {
        const {data, error: authError} = await supabase.auth.signUp({
          email: email.trim(),
          password,
          options: {emailRedirectTo: window.location.origin},
        });
        if (authError) throw authError;
        if (!data.session) {
          setMessage('Check your email to verify your account, then sign in.');
        }
      } else if (mode === 'sign-in') {
        const {error: authError} = await supabase.auth.signInWithPassword({
          email: email.trim(),
          password,
        });
        if (authError) throw authError;
      } else if (mode === 'reset') {
        const {error: authError} = await supabase.auth.resetPasswordForEmail(
          email.trim(),
          {redirectTo: window.location.origin},
        );
        if (authError) throw authError;
        setMessage('If that email has an account, a password-reset link has been sent.');
      } else {
        const {error: authError} = await supabase.auth.updateUser({password: newPassword});
        if (authError) throw authError;
        setMessage('Your password has been updated. You can continue using DocuMind.');
        onModeChange('sign-in');
      }
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : 'Authentication failed.');
    } finally {
      setBusy(false);
    }
  }

  const recovery = mode === 'recovery';
  const reset = mode === 'reset';
  const signingUp = mode === 'sign-up';
  const title = recovery
    ? 'Choose a new password'
    : reset
      ? 'Reset your password'
      : signingUp
        ? 'Create your account'
        : 'Welcome back';

  return (
    <main className="auth-page">
      <section className="auth-card">
        <div className="auth-brand">
          <div className="brand-mark"><BrainCircuit size={22}/></div>
          <div><b>DocuMind</b><small>DOCUMENT INTELLIGENCE</small></div>
        </div>
        <div className="auth-heading">
          <div className="eyebrow"><ShieldCheck size={14}/> PRIVATE WORKSPACE</div>
          <h1>{title}</h1>
          <p>Sign in to keep your uploads, summaries, and reports private to your account.</p>
        </div>
        <form onSubmit={submit}>
          {!recovery && (
            <label>
              Email address
              <span className="auth-input"><Mail size={16}/><input autoComplete="email" type="email" required value={email} onChange={event => setEmail(event.target.value)}/></span>
            </label>
          )}
          {mode === 'sign-in' || signingUp ? (
            <label>
              Password
              <input className="auth-password" autoComplete={signingUp ? 'new-password' : 'current-password'} type="password" minLength={8} required value={password} onChange={event => setPassword(event.target.value)}/>
            </label>
          ) : null}
          {recovery && (
            <label>
              New password
              <input className="auth-password" autoComplete="new-password" type="password" minLength={8} required value={newPassword} onChange={event => setNewPassword(event.target.value)}/>
            </label>
          )}
          {error && <div className="auth-alert error">{error}</div>}
          {message && <div className="auth-alert success">{message}</div>}
          <button className="primary auth-submit" disabled={busy}>
            {busy && <LoaderCircle className="spin" size={16}/>}
            {recovery ? 'Update password' : reset ? 'Send reset link' : signingUp ? 'Create account' : 'Sign in'}
          </button>
        </form>
        {mode === 'sign-in' && (
          <div className="auth-links">
            <button type="button" onClick={() => onModeChange('reset')}>Forgot password?</button>
            <span>New to DocuMind? <button type="button" onClick={() => onModeChange('sign-up')}>Create an account</button></span>
          </div>
        )}
        {mode === 'sign-up' && (
          <div className="auth-links"><span>Already registered? <button type="button" onClick={() => onModeChange('sign-in')}>Sign in</button></span></div>
        )}
        {(mode === 'reset' || recovery) && (
          <div className="auth-links"><button type="button" onClick={() => onModeChange('sign-in')}>Back to sign in</button></div>
        )}
        <p className="auth-footnote">Each account has a private job history. New accounts must verify their email.</p>
      </section>
    </main>
  );
}
