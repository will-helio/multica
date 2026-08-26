package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/multica-ai/multica/server/internal/middleware"
	"github.com/multica-ai/multica/server/internal/util"
	db "github.com/multica-ai/multica/server/pkg/db/generated"
)

// --- shared forgery-test fixtures -----------------------------------------
//
// These reproduce the actor-identity forgery vulnerability class: any
// authenticated workspace member holding a normal PAT/JWT (no X-Actor-Source:
// task_token) can send forged legacy X-Agent-ID + X-Task-ID headers and have
// resolveActor return ("agent", <id>) for an agent identity they do not
// control, because the legacy fallback path only checks that the named agent
// exists in the workspace and that the named task belongs to that agent --
// never that the CALLER is the agent process.
//
// forgeryTestActors seeds:
//   - ownerID: owns a PRIVATE agent (targetAgentID) that only the owner may
//     access/invoke.
//   - attackerID: a plain workspace member -- NOT the owner, NOT an admin,
//     NOT on any allow-list for the target agent.
//   - decoyAgentID: an unrelated agent that actually exists in the
//     workspace (attacker doesn't own or control it either -- forging only
//     requires that SOME agent+task pair resolve, per the vulnerability).
//   - decoyTaskID: a real agent_task_queue row for decoyAgentID whose
//     originator_user_id/accountable_user_id is ownerID -- this is the
//     "attacker-chosen task" that the confused-deputy path (cluster 3) reads
//     to borrow ownerID's invoke rights.
func forgeryTestActors(t *testing.T) (ownerID, attackerID, targetAgentID, decoyAgentID, decoyTaskID string) {
	t.Helper()
	ctx := context.Background()
	nonce := time.Now().UnixNano()

	mkUser := func(label string) string {
		var id string
		if err := testPool.QueryRow(ctx, `INSERT INTO "user" (name, email) VALUES ($1, $2) RETURNING id`,
			label, fmt.Sprintf("%s-%d@multica.test", label, nonce)).Scan(&id); err != nil {
			t.Fatalf("create user %s: %v", label, err)
		}
		t.Cleanup(func() { testPool.Exec(context.Background(), `DELETE FROM "user" WHERE id = $1`, id) })
		if _, err := testPool.Exec(ctx, `INSERT INTO member (workspace_id, user_id, role) VALUES ($1, $2, 'member')`,
			testWorkspaceID, id); err != nil {
			t.Fatalf("add member %s: %v", label, err)
		}
		return id
	}

	ownerID = mkUser("forgery-owner")
	attackerID = mkUser("forgery-attacker")

	runtimeID := handlerTestRuntimeID(t)

	if err := testPool.QueryRow(ctx, `
		INSERT INTO agent (
			workspace_id, name, description, runtime_mode, runtime_config,
			runtime_id, visibility, permission_mode, max_concurrent_tasks, owner_id,
			instructions, custom_env, custom_args
		)
		VALUES ($1, $2, '', 'cloud', '{}'::jsonb, $3, 'private', 'private', 1, $4, '', '{}'::jsonb, '[]'::jsonb)
		RETURNING id
	`, testWorkspaceID, fmt.Sprintf("forgery-target-agent-%d", nonce), runtimeID, ownerID).Scan(&targetAgentID); err != nil {
		t.Fatalf("create target private agent: %v", err)
	}
	t.Cleanup(func() { testPool.Exec(context.Background(), `DELETE FROM agent WHERE id = $1`, targetAgentID) })

	// Decoy agent: exists in the workspace, but is unrelated to the target.
	// It is owned by the attacker themselves -- forging identity does not even
	// require compromising someone else's agent.
	if err := testPool.QueryRow(ctx, `
		INSERT INTO agent (
			workspace_id, name, description, runtime_mode, runtime_config,
			runtime_id, visibility, permission_mode, max_concurrent_tasks, owner_id,
			instructions, custom_env, custom_args
		)
		VALUES ($1, $2, '', 'cloud', '{}'::jsonb, $3, 'private', 'private', 1, $4, '', '{}'::jsonb, '[]'::jsonb)
		RETURNING id
	`, testWorkspaceID, fmt.Sprintf("forgery-decoy-agent-%d", nonce), runtimeID, attackerID).Scan(&decoyAgentID); err != nil {
		t.Fatalf("create decoy agent: %v", err)
	}
	t.Cleanup(func() { testPool.Exec(context.Background(), `DELETE FROM agent WHERE id = $1`, decoyAgentID) })

	// A real task belonging to the decoy agent, attributed to ownerID as the
	// human originator (e.g. ownerID legitimately triggered the decoy agent
	// once). The attacker knows this task id (task ids are visible to any
	// workspace member via task listing endpoints).
	if err := testPool.QueryRow(ctx, `
		INSERT INTO agent_task_queue (
			agent_id, runtime_id, status, priority,
			originator_source, originator_user_id, accountable_user_id
		)
		VALUES ($1, $2, 'completed', 0, 'direct_human', $3, $3)
		RETURNING id
	`, decoyAgentID, runtimeID, ownerID).Scan(&decoyTaskID); err != nil {
		t.Fatalf("create decoy task: %v", err)
	}
	t.Cleanup(func() { testPool.Exec(context.Background(), `DELETE FROM agent_task_queue WHERE id = $1`, decoyTaskID) })

	return ownerID, attackerID, targetAgentID, decoyAgentID, decoyTaskID
}

// withWorkspaceCtxAs injects the workspace+member context (that the real chi
// middleware chain would set) for an arbitrary member, not just testUserID.
// Needed for handlers (e.g. CreateChatSession) that read workspace id from
// context rather than the X-Workspace-ID header.
func withWorkspaceCtxAs(t *testing.T, req *http.Request, userID string) *http.Request {
	t.Helper()
	memberRow, err := testHandler.Queries.GetMemberByUserAndWorkspace(context.Background(), db.GetMemberByUserAndWorkspaceParams{
		UserID:      util.MustParseUUID(userID),
		WorkspaceID: util.MustParseUUID(testWorkspaceID),
	})
	if err != nil {
		t.Fatalf("load member row for %s: %v", userID, err)
	}
	return req.WithContext(middleware.SetMemberContext(req.Context(), testWorkspaceID, memberRow))
}

// forgeRequestAs builds a request as attackerID carrying forged legacy
// X-Agent-ID / X-Task-ID headers -- exactly what the vulnerability requires:
// no X-Actor-Source (the middleware only ever sets that itself on the mat_
// task-token path; a request authenticated with a normal mul_ PAT / JWT never
// carries it), just a plain member identity plus the two legacy headers.
func forgeRequestAs(attackerID, method, path string, body any, forgedAgentID, forgedTaskID string) *http.Request {
	req := newRequestAs(attackerID, method, path, body)
	req.Header.Set("X-Agent-ID", forgedAgentID)
	req.Header.Set("X-Task-ID", forgedTaskID)
	return req
}

// TestGetAgent_ForgedLegacyHeaders_BypassPrivateAgentGate is the PHASE 1
// cluster-1 reproduction: a plain workspace member -- not the agent owner,
// not an admin -- forges X-Agent-ID/X-Task-ID for an agent+task pair they do
// not own, and resolveActor is tricked into returning ("agent", <id>).
// canAccessPrivateAgent's `if actorType == "agent" { return true }` branch
// (agent_access.go ~120) then admits them to a PRIVATE agent's detail view
// with NO check that actorID is the agent in question, or anything else.
//
// This test asserts the SECURE behavior (403). Today it is expected to FAIL
// -- i.e. the request actually succeeds with 200 -- proving the exploit.
func TestGetAgent_ForgedLegacyHeaders_BypassPrivateAgentGate(t *testing.T) {
	if testHandler == nil || testPool == nil {
		t.Skip("database not available")
	}

	_, attackerID, targetAgentID, decoyAgentID, decoyTaskID := forgeryTestActors(t)

	// Baseline: attacker with NO forged headers is correctly denied.
	w := httptest.NewRecorder()
	testHandler.GetAgent(w, withURLParam(newRequestAs(attackerID, "GET", "/api/agents/"+targetAgentID, nil), "id", targetAgentID))
	if w.Code != http.StatusForbidden {
		t.Fatalf("baseline (no forged headers): expected 403, got %d: %s", w.Code, w.Body.String())
	}

	// Exploit: attacker forges X-Agent-ID/X-Task-ID for an agent+task they
	// don't own or control (decoyAgentID/decoyTaskID), targeting a totally
	// unrelated private agent (targetAgentID) they have no access to.
	w = httptest.NewRecorder()
	req := forgeRequestAs(attackerID, "GET", "/api/agents/"+targetAgentID, nil, decoyAgentID, decoyTaskID)
	testHandler.GetAgent(w, withURLParam(req, "id", targetAgentID))

	if w.Code != http.StatusForbidden {
		t.Errorf("VULNERABLE (cluster 1, agent_access.go canAccessPrivateAgent): "+
			"plain member forged X-Agent-ID=%s X-Task-ID=%s and read PRIVATE agent %s "+
			"they have no legitimate access to -- expected 403, got %d: %s",
			decoyAgentID, decoyTaskID, targetAgentID, w.Code, w.Body.String())
	}
}

// TestCreateChatSession_ForgedLegacyHeaders_BorrowsOriginatorToInvoke is the
// PHASE 1 cluster-3 reproduction: canInvokeAgent's confused-deputy path.
// When actorType != "member", canInvokeAgent judges the *originator* of the
// attacker-CHOSEN X-Task-ID task (invokeOriginatorFromRequest,
// agent_access.go ~182), not the caller. A plain member with forged headers
// can borrow the originator identity of ANY task they know the id of --
// here, a task on an unrelated decoy agent whose originator happens to be
// the owner of a private agent the attacker wants to invoke -- and
// CreateChatSession (chat.go ~88) will enqueue a run against it.
//
// This test asserts the SECURE behavior (403, no session created). Today it
// is expected to FAIL -- i.e. the request actually succeeds with 201 and
// creates a chat session/run -- proving the exploit.
func TestCreateChatSession_ForgedLegacyHeaders_BorrowsOriginatorToInvoke(t *testing.T) {
	if testHandler == nil || testPool == nil {
		t.Skip("database not available")
	}

	_, attackerID, targetAgentID, decoyAgentID, decoyTaskID := forgeryTestActors(t)

	countSessions := func() int {
		var n int
		if err := testPool.QueryRow(context.Background(),
			`SELECT count(*) FROM chat_session WHERE agent_id = $1`, targetAgentID).Scan(&n); err != nil {
			t.Fatalf("count chat sessions: %v", err)
		}
		return n
	}
	before := countSessions()

	// Baseline: attacker with no forged headers is correctly denied invoke
	// on a private agent they don't own.
	w := httptest.NewRecorder()
	baselineReq := withWorkspaceCtxAs(t, newRequestAs(attackerID, "POST", "/api/chat/sessions", map[string]any{
		"agent_id": targetAgentID,
		"title":    "forgery baseline",
	}), attackerID)
	testHandler.CreateChatSession(w, baselineReq)
	if w.Code != http.StatusForbidden {
		t.Fatalf("baseline (no forged headers): expected 403, got %d: %s", w.Code, w.Body.String())
	}

	// Exploit: attacker forges X-Agent-ID/X-Task-ID pointing at the decoy
	// task (originator = the target agent's owner) to invoke the private
	// target agent they are not entitled to invoke.
	w = httptest.NewRecorder()
	req := forgeRequestAs(attackerID, "POST", "/api/chat/sessions", map[string]any{
		"agent_id": targetAgentID,
		"title":    "forgery exploit",
	}, decoyAgentID, decoyTaskID)
	req = withWorkspaceCtxAs(t, req, attackerID)
	testHandler.CreateChatSession(w, req)

	if w.Code == http.StatusCreated {
		var session struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &session); err == nil && session.ID != "" {
			t.Cleanup(func() { testPool.Exec(context.Background(), `DELETE FROM chat_session WHERE id = $1`, session.ID) })
		}
	}

	if w.Code != http.StatusForbidden {
		t.Errorf("VULNERABLE (cluster 3, agent_access.go canInvokeAgent/invokeOriginatorFromRequest): "+
			"plain member forged X-Agent-ID=%s X-Task-ID=%s (originator=agent owner) and invoked "+
			"PRIVATE agent %s they are not entitled to invoke -- expected 403, got %d: %s",
			decoyAgentID, decoyTaskID, targetAgentID, w.Code, w.Body.String())
	}
	after := countSessions()
	if after != before && w.Code == http.StatusCreated {
		t.Errorf("VULNERABLE: forged invoke persisted a real chat_session for agent %s (before=%d after=%d)",
			targetAgentID, before, after)
	}
}
