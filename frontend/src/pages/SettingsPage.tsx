import { useEffect, useState } from 'react';
import { SettingsAPI } from '../api/client';

export default function SettingsPage() {
  const [data, setData] = useState<Record<string, string> | null>(null);

  useEffect(() => {
    SettingsAPI.get().then(setData);
  }, []);

  return (
    <div>
      <h2>Settings</h2>
      <div className="card">
        <p>Current backend providers (read-only; configure via backend <code>.env</code>):</p>
        {data && (
          <table>
            <tbody>
              {Object.entries(data).map(([k, v]) => (
                <tr key={k}>
                  <td><code>{k}</code></td>
                  <td>{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
