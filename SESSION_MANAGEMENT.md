# Session Management

The system has been extended with advanced user session management, enabling server-side conversation context storage.

## Features

### Session Management
- **Automatic session creation** - if no `thread_id` is provided, the system creates a new session
- **User tracking** - optional binding of sessions to `user_id`
- **Context storage** - each session stores history of queries and responses
- **Automatic cleanup** - expired sessions are automatically removed
- **Session statistics** - monitoring of activity and usage

### Storage Types
- **Memory** - in-memory storage (default, fast, data lost on restart)
- **SQLite** - persistent database storage (data preserved after restart)

## Configuration

Add the `sessions` section to the `config.yaml` file:

```yaml
sessions:
  storage_type: memory  # memory or sqlite
  db_path: data/sessions.db  # path to SQLite database
  timeout_hours: 24  # session expiration after X hours
  cleanup_interval_minutes: 60  # cleanup interval in minutes
  max_sessions_per_user: 10  # maximum number of sessions per user
```

### Parameter Description:
- `storage_type`: Storage type (`memory` or `sqlite`)
- `db_path`: Path to SQLite database file (only for `sqlite`)
- `timeout_hours`: Number of hours after which an inactive session expires
- `cleanup_interval_minutes`: How often to run cleanup (in minutes)
- `max_sessions_per_user`: Session limit per user (0 = no limit)

## API Endpoints

### Queries with Sessions

**POST /query**
```json
{
  "query": "Your question",
  "agent_type": "advanced",
  "thread_id": "session_123",  // optional - if missing, creates new session
  "user_id": "user_456",       // optional - user identifier
  "parameters": {}
}
```

### Session Management

**POST /sessions** - Create new session
```json
{
  "user_id": "user_123",  // optional
  "metadata": {           // optional
    "source": "webapp",
    "version": "1.0"
  }
}
```

**GET /sessions/{session_id}** - Get session information

**DELETE /sessions/{session_id}** - Delete session

**GET /users/{user_id}/sessions** - List user sessions

**GET /sessions/stats** - Session statistics

**POST /sessions/cleanup** - Manual cleanup trigger

## Usage Examples

### 1. Creating Session and Conversation

```python
import httpx

async def chat_example():
    async with httpx.AsyncClient() as client:
        # Create session
        response = await client.post("http://localhost:8080/sessions", json={
            "user_id": "john_doe",
            "metadata": {"source": "chat_app"}
        })
        session_id = response.json()["session_id"]
        
        # First question
        response = await client.post("http://localhost:8080/query", json={
            "query": "I am a Python programmer",
            "thread_id": session_id,
            "user_id": "john_doe"
        })
        
        # Second question in the same session
        response = await client.post("http://localhost:8080/query", json={
            "query": "What tools do you recommend for work?",
            "thread_id": session_id,
            "user_id": "john_doe"
        })
        # Agent will have context from the previous question
```

### 2. Automatic Session Creation

```python
# If you don't provide thread_id, the system will create a new session automatically
response = await client.post("http://localhost:8080/query", json={
    "query": "Hello, how are you?",
    "user_id": "john_doe"
})

# session_id can be found in the response
session_id = response.json()["metadata"]["thread_id"]
```

### 3. Session Monitoring

```python
# Statistics
stats = await client.get("http://localhost:8080/sessions/stats")
print(f"Active sessions: {stats.json()['active_sessions_memory']}")

# User sessions
user_sessions = await client.get("http://localhost:8080/users/john_doe/sessions")
for session in user_sessions.json()["sessions"]:
    print(f"Session: {session['session_id']}, Queries: {session['metadata'].get('query_count', 0)}")
```

## Testing

Start the server:
```bash
python main.py server --port 8080
```

Run session tests:
```bash
python tests/test_sessions.py
```

## Session Structure

Each session contains:

### Metadata
- `query_count` - number of queries in session
- `last_query` - last query
- `last_agent_type` - last used agent
- `created_from` - where the session was created from
- user data (custom metadata)

### Context
- `last_query` - last query (full)
- `last_response` - last response (truncated)
- `last_execution_time` - execution time of last query
- `last_agent_used` - name of last agent
- `error_count` - number of errors in session
- `last_error` - last error

## Security

- Sessions are automatically cleaned up after expiration
- Session limit per user prevents excessive memory usage
- SQLite uses safe parameterized queries
- No automatic session sharing between users

## Migration from Previous Version

Existing queries will work without changes:
- If you don't provide `thread_id`, a new session will be created
- If you provide a non-existent `thread_id`, a new session will be created with that ID
- All parameters are optional and backward-compatible

## Troubleshooting

### Problem: Sessions are not preserved after restart
**Solution**: Change `storage_type` to `sqlite` in configuration

### Problem: Too many sessions in memory
**Solution**: Decrease `timeout_hours` or `max_sessions_per_user`

### Problem: Database error
**Solution**: Check if `data/` directory exists and has write permissions

### Problem: Session not found
**Solution**: Check if `session_id` is correct and session hasn't expired 