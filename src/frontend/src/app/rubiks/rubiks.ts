
// 633 reprensentation
export type side = number[];
export type cube = side[];
// Fancy reprensentation
export type cube20 = number[];

export interface IResponse {
	state: cube;
	state20: cube20;
}

export interface IActionRequest {
	action: number;
	state20: cube20;
}

