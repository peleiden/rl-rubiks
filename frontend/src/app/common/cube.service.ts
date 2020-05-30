import { Injectable } from '@angular/core';
import { cube, cube69, action } from './rubiks';

import maps from "../../assets/maps.json";

export function deepCopy(obj: any) {
  if (Array.isArray(obj)) {
    return obj.map(val => deepCopy(val));
  } else if (obj && typeof obj === "object") {
    const newObj = {};
    for (let key of Object.keys(obj)) {
      newObj[key] = deepCopy(newObj[key]);
    }
    return newObj;
  } else {
    return obj;
  }
}

@Injectable({
  providedIn: 'root'
})
export class CubeService {

  /*
  Cube environment methods
  See librubiks/cube/cube.py for documentation, as this implementation is nearly identical to that found in Cube for 20 x 24 representation
  */

  actionSpace: action[] = Array(12).fill(null).map((val, i) => [Math.floor(i/2), i % 2 == 0]);
  actionDim = this.actionSpace.length;

  solvedState: cube = [0, 3, 6, 9, 12, 15, 18, 21, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
  private _currentState: cube = deepCopy(this.solvedState);
  currentState69: cube69 = this.as69(this._currentState);

  actions = ["f", "F", "b", "B", "t", "T", "d", "D", "l", "L", "r", "R"];

  get currentState() {
    return deepCopy(this._currentState);
  }

  set currentState(state: cube) {
    this._currentState = state;
    this.currentState69 = this.as69(this._currentState);
  }

  private isSolved(): boolean {
    return this.currentState === this.getSolved();
  }
  
  private getSolved(): cube {
    return deepCopy(this.solvedState);
  }

  public getAction(actionIndex: number): action {
    return this.actionSpace[actionIndex];
  }

  public reset() {
    this.currentState = deepCopy(this.solvedState);
    this.currentState69 = this.as69(this.currentState);
  }

  public inplaceRotate(face: number, pos_rev: boolean) {
    this.currentState = this.rotate(this.currentState, face, pos_rev);
  }

  public rotate(state: cube, face: number, pos_rev: boolean) {
    const map = pos_rev ? maps.map_pos[face] : maps.map_neg[face];
    const alteredState = deepCopy(state);
    for (let i = 0; i < 8; i ++) {
      alteredState[i] += map[0][state[i]];
    }
    for (let i = 8; i < 20; i ++) {
      alteredState[i] += map[1][state[i]];
    }
    return alteredState;
  }

  public scramble(moves: number): number[] {
    // Scrambles from the current state
    let state = this.currentState;
    const actions: number[] = [];
    for (let i = 0; i < moves; i ++) {
      actions.push(Math.floor(Math.random() * this.actionDim));
    }
    return actions;
  }

  public as69(state: cube) {
    const state633 = Array(6).fill(0).map((val, i) => [[i, i, i], [i, i, i], [i, i, i]]);  // i
    for (let i = 0; i < 8; i ++) {  // Corners
      const pos = Math.floor(state[i] / 3);
      let orientation = state[i] % 3;
      if ([0, 2, 5, 7].includes(pos)) {
        orientation *= -1;
      }
      const values = this.roll(maps.corner_633maps[i].map(val => val[0]), orientation);
      const corner_map = maps.corner_633maps[pos];
      state633[corner_map[0][0]][corner_map[0][1]][corner_map[0][2]] = values[0];
      state633[corner_map[1][0]][corner_map[1][1]][corner_map[1][2]] = values[1];
      state633[corner_map[2][0]][corner_map[2][1]][corner_map[2][2]] = values[2];
    }
    
    for (let i = 0; i < 12; i ++) {  // Sides
      const pos = Math.floor(this.currentState[i+8] / 2);
      const orientation = state[i+8] % 2;
      const values = this.roll(maps.side_633maps[i].map(val => val[0]), orientation);
      const side_map = maps.side_633maps[pos];
      state633[side_map[0][0]][side_map[0][1]][side_map[0][2]] = values[0];
      state633[side_map[1][0]][side_map[1][1]][side_map[1][2]] = values[1];
    }
    
    const state69 = [];
    for (const face of state633) {  // Reshape into 6x9 representation
      state69.push([...face[0], ...face[1], ...face[2]]);
    }
    return state69;
  }

  private roll(array: any[], times: number): any[] {
    let shifted = deepCopy(array);
    if (times > 0) {
      while (times !== 0) {
        times --;
        const elem = shifted.pop();
        shifted = [elem, ...shifted];
      }
    } else if (times < 0) {
      while (times !== 0) {
        times ++;
        const elem = shifted.shift();
        shifted.push(elem)
      }
    }
    return shifted;
  }
}
