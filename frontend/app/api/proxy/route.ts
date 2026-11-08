import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const endpoint = req.nextUrl.searchParams.get("endpoint") || "solar_simulate";

    const pythonUrl = process.env.PYTHON_BACKEND_URL || "http://127.0.0.1:5000";

    const res = await fetch(`${pythonUrl}/api/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`Python service responded with status ${res.status}`);
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: unknown) {
    // If Python service is offline, calculate using pure TypeScript physics equations
    const message = err instanceof Error ? err.message : "Error connecting to AI service";
    return NextResponse.json({ fallback: true, error: message }, { status: 502 });
  }
}

