"""
Session Manager for Agent Application
Handles user sessions with persistent storage and metadata tracking
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import sqlite3
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager

@dataclass
class SessionInfo:
    """Information about a user session"""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = None
    last_accessed: datetime = None
    metadata: Dict[str, Any] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.context is None:
            self.context = {}


class SessionManager:
    """Manages user sessions with optional persistent storage"""
    
    def __init__(self, 
                 storage_type: str = "memory",
                 db_path: str = "sessions.db",
                 session_timeout_hours: int = 24,
                 cleanup_interval_minutes: int = 60,
                 max_sessions_per_user: int = 10):
        """
        Initialize session manager
        
        Args:
            storage_type: "memory" or "sqlite" for persistence
            db_path: Path to SQLite database file (if using sqlite)
            session_timeout_hours: Hours after which inactive sessions expire
            cleanup_interval_minutes: Interval for automatic cleanup
            max_sessions_per_user: Maximum sessions per user
        """
        self.storage_type = storage_type
        self.db_path = db_path
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self.max_sessions_per_user = max_sessions_per_user
        
        # In-memory storage
        self._sessions: Dict[str, SessionInfo] = {}
        self._lock = threading.RLock()
        
        # Setup persistent storage if needed
        if storage_type == "sqlite":
            self._init_sqlite()
        
        # Start cleanup thread
        self._last_cleanup = datetime.now()
        self._setup_cleanup()
    
    def _init_sqlite(self):
        """Initialize SQLite database for persistent storage"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # Only create directory if path contains directory component
            os.makedirs(db_dir, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TEXT,
                    last_accessed TEXT,
                    metadata TEXT,
                    context TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON sessions(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON sessions(last_accessed)
            """)
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper handling"""
        if self.storage_type != "sqlite":
            yield None
            return
            
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()
    
    def _setup_cleanup(self):
        """Setup automatic cleanup of expired sessions"""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval.total_seconds())
                try:
                    self.cleanup_expired_sessions()
                except Exception as e:
                    print(f"Session cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def create_session(self, user_id: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session
        
        Args:
            user_id: Optional user identifier
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session_id = f"session_{int(time.time() * 1000)}_{hash(time.time()) % 10000}"
        
        # Cleanup old sessions for user if limit exceeded
        if user_id and self.max_sessions_per_user > 0:
            self._cleanup_user_sessions(user_id)
        
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._sessions[session_id] = session_info
            
            if self.storage_type == "sqlite":
                self._save_session_to_db(session_info)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session information
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionInfo or None if not found
        """
        with self._lock:
            # Check memory first
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_accessed = datetime.now()
                
                if self.storage_type == "sqlite":
                    self._save_session_to_db(session)
                
                return session
            
            # Load from database if using sqlite
            if self.storage_type == "sqlite":
                session = self._load_session_from_db(session_id)
                if session:
                    session.last_accessed = datetime.now()
                    self._sessions[session_id] = session
                    self._save_session_to_db(session)
                    return session
        
        return None
    
    def update_session_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """
        Update session context
        
        Args:
            session_id: Session identifier
            context: Context data to update
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        with self._lock:
            session.context.update(context)
            session.last_accessed = datetime.now()
            
            if self.storage_type == "sqlite":
                self._save_session_to_db(session)
        
        return True
    
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update session metadata
        
        Args:
            session_id: Session identifier
            metadata: Metadata to update
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        with self._lock:
            session.metadata.update(metadata)
            session.last_accessed = datetime.now()
            
            if self.storage_type == "sqlite":
                self._save_session_to_db(session)
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            deleted = False
            
            if session_id in self._sessions:
                del self._sessions[session_id]
                deleted = True
            
            if self.storage_type == "sqlite":
                with self._get_db_connection() as conn:
                    if conn:
                        cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                        if cursor.rowcount > 0:
                            deleted = True
                        conn.commit()
        
        return deleted
    
    def list_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """
        List all sessions for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of SessionInfo objects
        """
        sessions = []
        
        with self._lock:
            # Check memory
            for session in self._sessions.values():
                if session.user_id == user_id:
                    sessions.append(session)
            
            # Load from database if using sqlite
            if self.storage_type == "sqlite":
                with self._get_db_connection() as conn:
                    if conn:
                        cursor = conn.execute(
                            "SELECT * FROM sessions WHERE user_id = ? ORDER BY last_accessed DESC",
                            (user_id,)
                        )
                        for row in cursor.fetchall():
                            session_id = row[0]
                            if session_id not in self._sessions:
                                session = self._row_to_session(row)
                                sessions.append(session)
        
        return sessions
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.now() - self.session_timeout
        cleaned_count = 0
        
        with self._lock:
            # Clean memory
            expired_sessions = [
                session_id for session_id, session in self._sessions.items()
                if session.last_accessed < cutoff_time
            ]
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
                cleaned_count += 1
            
            # Clean database
            if self.storage_type == "sqlite":
                with self._get_db_connection() as conn:
                    if conn:
                        cursor = conn.execute(
                            "DELETE FROM sessions WHERE last_accessed < ?",
                            (cutoff_time.isoformat(),)
                        )
                        cleaned_count += cursor.rowcount
                        conn.commit()
        
        if cleaned_count > 0:
            print(f"Cleaned up {cleaned_count} expired sessions")
        
        return cleaned_count
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics
        
        Returns:
            Dictionary with session statistics
        """
        with self._lock:
            active_sessions = len(self._sessions)
            
            # Count total sessions if using database
            total_sessions = active_sessions
            if self.storage_type == "sqlite":
                with self._get_db_connection() as conn:
                    if conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                        total_sessions = cursor.fetchone()[0]
            
            # Count by user
            users_count = len(set(s.user_id for s in self._sessions.values() if s.user_id))
            
            return {
                "active_sessions_memory": active_sessions,
                "total_sessions": total_sessions,
                "unique_users": users_count,
                "storage_type": self.storage_type,
                "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
                "last_cleanup": self._last_cleanup.isoformat()
            }
    
    def _cleanup_user_sessions(self, user_id: str):
        """Clean up excess sessions for a user"""
        user_sessions = self.list_user_sessions(user_id)
        if len(user_sessions) >= self.max_sessions_per_user:
            # Sort by last accessed and remove oldest
            user_sessions.sort(key=lambda s: s.last_accessed)
            sessions_to_remove = user_sessions[:-self.max_sessions_per_user + 1]
            
            for session in sessions_to_remove:
                self.delete_session(session.session_id)
    
    def _save_session_to_db(self, session: SessionInfo):
        """Save session to database"""
        with self._get_db_connection() as conn:
            if conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, user_id, created_at, last_accessed, metadata, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.created_at.isoformat(),
                    session.last_accessed.isoformat(),
                    json.dumps(session.metadata),
                    json.dumps(session.context)
                ))
                conn.commit()
    
    def _load_session_from_db(self, session_id: str) -> Optional[SessionInfo]:
        """Load session from database"""
        with self._get_db_connection() as conn:
            if conn:
                cursor = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_session(row)
        return None
    
    def _row_to_session(self, row) -> SessionInfo:
        """Convert database row to SessionInfo"""
        return SessionInfo(
            session_id=row[0],
            user_id=row[1],
            created_at=datetime.fromisoformat(row[2]),
            last_accessed=datetime.fromisoformat(row[3]),
            metadata=json.loads(row[4]) if row[4] else {},
            context=json.loads(row[5]) if row[5] else {}
        ) 