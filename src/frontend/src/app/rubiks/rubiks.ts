
// 633 reprensentation
export type side = number[];
export type cube = side[];
// Fancy reprensentation
export type cube20 = number[];


export interface IActionRequest {
	action: number;
	state20: cube20;
}

export interface IScrambleRequest {
	depth: number;
	state20: cube20;
}

export interface ICubeResponse {
	state: cube;
	state20: cube20;
}

export interface IActionsResponse {
	states: cube[];
	finalState20: cube20;
}


