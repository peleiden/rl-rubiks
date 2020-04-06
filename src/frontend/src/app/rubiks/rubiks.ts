
// 633 reprensentation
export type side = number[];
export type cube = side[];
// Fancy reprensentation
export type cube20 = number[];

export interface IInfoResponse {
	cuda: boolean;
	agents: string[];
}

export interface ICubeResponse {
	state: cube;
	state20: cube20;
}

export interface IActionRequest {
	action: number;
	state20: cube20;
}

export interface IScrambleRequest {
	depth: number;
	state20: cube20;
}

export interface IScrambleResponse {
	states: cube[];
	finalState20: cube20;
}

export interface ISolveRequest {
	agentIdx: number;
	timeLimit: number;
	state20: cube20;
}

export interface ISolveResponse {
	solution: boolean;
	actions: number[];
	states: cube[];
	finalState20: cube20;
}


